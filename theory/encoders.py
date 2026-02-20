import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import math


class MLP(nn.Module):
    def __init__(self, hparams, act_fun = nn.ReLU, use_bias=True):
        super().__init__()

        self.layer_dim = hparams['layer_dim']
        self.input_dim = hparams['input_dim']
        self.block_grad = hparams['layerwise']
        self.block_layers = hparams['block_layers']
        self.batch_norm = False
        if 'batch_norm' in hparams:
            self.batch_norm = hparams['batch_norm']
        self.preact_ind = -3 if self.batch_norm else -2
        if isinstance(self.block_layers, int):
            self.block_layers = [hparams['block_layers']] * (len(self.layer_dim))
        in_dim = self.input_dim
        self.blocks = nn.ModuleList()
        self.block_dims = []
        for i, dim in enumerate(self.layer_dim):
            layer = []
            layer.append(nn.Linear(in_dim, dim, bias=use_bias))
            self.block_dims.append(dim)
            if (i < len(self.layer_dim) - 1) or hparams['last_layer_activation']:
                layer.append(act_fun())
            else:
                layer.append(nn.Identity())
            if self.batch_norm:
                layer.append(nn.BatchNorm1d(dim, affine=False))
            for j in range(1, self.block_layers[i]):
                layer.append(nn.Linear(dim, dim, bias=use_bias))
                self.block_dims.append(dim)
                layer.append(act_fun())
                if self.batch_norm:
                    layer.append(nn.BatchNorm1d(dim))
            in_dim = dim
            self.blocks.append(nn.Sequential(*layer))
        

    def forward(self, x):

        if self.block_grad:
            block_pass = torch.detach
        else:
            def block_pass(v): return v
        
        all_x = []
        for i, layer in enumerate(self.blocks):
            if i == 0:
                x = layer(x)
            else:
                x = layer(block_pass(x))
            #x.requires_grad = True
            all_x.append(x)
        return all_x 
    
    def forward_with_preact(self, x):

        if self.block_grad:
            block_pass = torch.detach
        else:
            def block_pass(v): return v

        all_x = [x]
        all_preact = []
        for i, layer in enumerate(self.blocks):
            if i == 0:
                preact = layer[:self.preact_ind+1](x)
                x = layer[self.preact_ind+1:](preact)
            else:
                preact = layer[:self.preact_ind+1](block_pass(x))
                x = layer[self.preact_ind+1:](preact)
            #x.requires_grad = True
            all_x.append(x)
            all_preact.append(preact)
        return all_x, all_preact


def custom_init(layer, strategy):
    if strategy == 'default':
        return
    elif strategy[:4] == 'uni_':
        scale = math.sqrt(layer.weight.shape[1])
        bound = float(strategy[4:])/scale
        print('initialize with bound {}'.format(bound))
        nn.init.uniform_(layer.weight, a=-bound, b=bound)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, a=-bound, b=bound)
    elif strategy == 'zero':
        nn.init.zeros_(layer.weight)
    elif strategy == 'zero_all':
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif strategy == 'kaiming':
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    elif strategy == 'orthogonal':
        nn.init.orthogonal_(layer.weight)
    elif strategy == 'xavier':
        gain = nn.init.calculate_gain('relu', 0.2)  
        nn.init.xavier_normal_(layer.weight, gain)
    else:
        raise NotImplementedError('strategy not implemented')

class CNN(nn.Module):
    def __init__(self, hparams, pooling=True, act_fun = nn.ReLU, use_bias=True):
        super().__init__()

        self.layer_dim = hparams['layer_dim']
        self.input_channel = 1
        self.block_grad = hparams['layerwise']
        self.strides = hparams['conv_strides']
        
        in_dim = self.input_channel
        self.pool = pooling
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.ModuleList([])
        for i, dim in enumerate(self.layer_dim):
            layer = []
            layer.append(('conv', nn.Conv2d(in_dim, dim, 2, self.strides[i], 0, bias=use_bias)))
            layer.append(('relu', act_fun()))
            if hparams['max_pool'][i] > 1:
                layer.append(('pool', nn.MaxPool2d(hparams['max_pool'][i], hparams['max_pool'][i])))
            in_dim = dim
            self.layers.append(nn.Sequential(OrderedDict(layer)))

    def forward(self, x):

        if self.block_grad:
            block_pass = torch.detach
        else:
            def block_pass(v): return v
        
        all_x = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
            else:
                x = layer(block_pass(x))
            if self.pool:
                x_out = self.average_pool(x).squeeze()
            else:
                x_out = x
            all_x.append(x_out)
        return all_x 

