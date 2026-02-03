import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

class triangle(nn.Module):
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)


def stdnorm (x, dims = [1,2,3]):
    x = x - torch.mean(x, dim=(dims), keepdim=True) / (1e-10 + torch.std(x, dim=(dims), keepdim=True))
    return x


class VGG(nn.Module):
    def __init__(
        self,
        opt,
        block_idx,
        blocks,
        in_channels,
        x_dim,
        kernel_=3,
        stride_=1,
        padding_=1, 
        calc_loss=False,
    ):
        super(VGG, self).__init__()
        self.encoder_num = block_idx
        self.opt = opt
        self.block_idx = block_idx
        self.peer_norm = 1.0
        self.momentum = 0.9
        

        self.save_vars = self.opt.save_vars_for_update_calc == block_idx+1

        # Layer
        if self.opt.custom_mlp is not None:
            self.mlp_range = [int(v) for v in self.opt.custom_mlp.split('-')]
        else:
            self.mlp_range = []

        if self.opt.input_decorr_layer is not None:
            self.input_decorr_range = [int(v) for v in self.opt.input_decorr_layer.split('-')]
            print('Layers to add input decorrelation {}'.format(self.input_decorr_range))
        else:
            self.input_decorr_range = []
        
        # if block_idx == 0:
        #     kernel_ = 5
        #     padding_ = 2
        self.model, self.out_dim = self.make_layers(blocks[block_idx], block_idx, in_channels, kernel_, stride_, padding_, x_dim)
        #self.model = self.make_layers_new(block_idx, opt.config['layer_configs'][block_idx])

        # Params
        self.calc_loss = calc_loss


        def get_last_index(block):
            if block[-1] == 'M':
                last_ind = -2
            else:
                last_ind = -1
            return last_ind

        self.last_ind = get_last_index(blocks[block_idx])
        self.in_planes = blocks[block_idx][self.last_ind]
        self.layer_activity_dim = self.in_planes
        if self.opt.triangle_act:
            self.triangle = triangle()

    def get_output_dim(self):
        return self.out_dim
        
    def constrain_weight(self, method='constrain'):
        weight = self.model[0].weight
        bias = self.model[0].bias
        if method == 'constrain':
            weight_norm = torch.norm(weight, dim = [0,1])
            return torch.square(weight_norm - 1).mean() + torch.square(bias).mean()
        elif method == 'l2':
            return torch.square(weight).sum() + torch.square(bias).sum()
        else:
            raise ValueError('method of constraining weights not recognized')

    def make_layers(self, block, block_idx, in_channels, kernel_, stride_, padding_, x_dim, inplace=False):
        # x_dim in shape of (B, C, H, W)
        layers = []
        for i, v in enumerate(block):
            if v == 'M':
                if not self.opt.use_stride:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    x_dim = (x_dim[0], x_dim[1], x_dim[2]//2, x_dim[3]//2)
            else:
                if i < len(block) - 1 and self.opt.use_stride and block[i+1] == 'M':
                    stride_ = 2
                elif i in self.mlp_range: # only valid for stride_=1
                    kernel_ = 1
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_, stride = stride_, padding=padding_, padding_mode = self.opt.padding_mode)
                
                if self.opt.batch_norm and block_idx>0:
                    layers += [nn.BatchNorm2d(in_channels), conv2d, nn.ReLU(inplace=inplace)]
                elif self.opt.layer_norm and block_idx>0:
                    layers += [nn.LayerNorm(x_dim[1:]), conv2d, nn.ReLU(inplace=inplace)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=inplace)]

                x_dim = (x_dim[0], v, int((x_dim[2] + 2*padding_ - kernel_)/stride_ + 1), int((x_dim[3] + 2*padding_ - kernel_)/stride_ + 1))
                in_channels = v


        return nn.Sequential(*layers), x_dim
    
    def make_layers_new(self, block_idx, config, batch_norm=False, inplace=False):

        conv2d = nn.Conv2d(config['ch_in'], config['channels'], kernel_size=config['kernel_size'], stride=1, padding=config['pad'], padding_mode=config['padding_mode'])
        layers = [conv2d]

        if config['pooltype'] == 'Max':
            layers.append(nn.MaxPool2d(kernel_size=config['pool_size'], stride=config['pool_stride'], padding=config['pool_pad']))
        elif config['pooltype'] == 'Avg':
            layers.append(nn.AvgPool2d(kernel_size=config['pool_size'], stride=config['pool_stride'], padding=config['pool_pad']))
        elif config['pooltype'] != 'No':
            raise NotImplementedError('pooltype {} not implemented'.format(config['pooltype']))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(config['channels']))

        if config['act'] == 'relu':
            layers.append(nn.ReLU(inplace=inplace))
        elif config['act'] == 'triangle':
            layers.append(triangle())
        else:
            raise NotImplementedError('activation {} not implemented'.format(config['act']))

        return nn.Sequential(*layers)
    
    
    def forward(self, model_input, label=None, eval=False, module_layer=-1):
        # if self.opt.std_norm:
        #     model_input = stdnorm(model_input, dims=[1,2,3])
        if module_layer != -1:
            return self.model[:module_layer](model_input)
        else:
            preact = self.model[0](model_input)
            z = self.model[1:](preact)
            if self.opt.triangle_act:
                h = self.model[2:](self.triangle(preact))
                return h, z
            else:
                return z, z
    


class SCFF(nn.Module):
    def __init__(
        self,
        opt,
        block_idx,
        blocks,
        in_channels,
        kernel_=3,
        stride_=1,
        padding_=1,
        pool_kernel=4,
        pool_stride=2,
        calc_loss=False,
        act='relu'
    ):
        super(SCFF, self).__init__()
        self.encoder_num = block_idx
        self.opt = opt
        self.block_idx = block_idx
        self.peer_norm = 1.0
        self.momentum = 0.9
        self.act = triangle() if act == 'triangle' else nn.ReLU()
        

        self.save_vars = self.opt.save_vars_for_update_calc == block_idx+1

        # Layer
        if self.opt.custom_mlp is not None:
            self.mlp_range = [int(v) for v in self.opt.custom_mlp.split('-')]
        else:
            self.mlp_range = []

        if self.opt.input_decorr_layer is not None:
            self.input_decorr_range = [int(v) for v in self.opt.input_decorr_layer.split('-')]
            print('Layers to add input decorrelation {}'.format(self.input_decorr_range))
        else:
            self.input_decorr_range = []

        self.model = self.make_layers(blocks[block_idx], block_idx, in_channels, kernel_, stride_, padding_, pool_kernel, pool_stride)

        # Params
        self.calc_loss = calc_loss


        self.in_planes = blocks[block_idx][0]
        self.layer_activity_dim = self.in_planes
        
        

    def make_layers(self, block, block_idx, in_channels, kernel_, stride_, padding_, pool_kernel, pool_stride, batch_norm=False, inplace=False):
        layers = []
        for i, v in enumerate(block):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding = padding_)]
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding = padding_)]
            else:
                if i < len(block) - 1 and self.opt.use_stride and block[i+1] == 'M':
                    stride_ = 2
                elif i in self.mlp_range: # only valid for stride_=1
                    kernel_ = 1
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_, stride = stride_, padding=int((kernel_-1)/2), padding_mode = self.opt.padding_mode)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), self.act]
                else:
                    layers += [conv2d, self.act]
                in_channels = v

        return nn.Sequential(*layers)
    
    def forward(self, model_input, label=None, eval=False, module_layer=-1):
        if module_layer != -1:
            return self.model[:module_layer](model_input)
        else:
            return self.model(model_input)
    
    