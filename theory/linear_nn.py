import torch
import numpy as np
import torch.nn as nn
from encoders import MLP, CNN, custom_init
from loss import ClappFlexLoss
from utils import merge_in_dict, compute_cos_sim, compute_mse
from tqdm import tqdm


class ClappMLP(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.patch_input = hparams['patch_input']
        if self.patch_input:
            hparams['input_dim'] = hparams['patch_size'] ** 2
            self.patch_size = hparams['patch_size']
        self.encoder = MLP(hparams, act_fun=nn.Identity if hparams['linear_nn'] else nn.ReLU, use_bias=False)
        self.block_grad = hparams['layerwise']
        self.layer_dim = hparams['layer_dim'] if self.block_grad else [hparams['layer_dim'][-1]]
        self.device = hparams['device']
        self.loss = ClappFlexLoss(hparams)
        self.flatten = nn.Flatten()
        self.random_label = None
        self.preact_index = 0
        self.fb_idx = hparams['fb_idx']
        self.custom_init = hparams['custom_init']
        if self.custom_init is not None:
            self.init_mlp()

    def init_mlp(self):
        if self.custom_init == 'default':
            return
        else:
            for l_seq in self.encoder.blocks:
                custom_init(l_seq[0], self.custom_init)
                print('{} weight initialized'.format(self.custom_init))
            return

    def forward(self, x, return_h=False):
        h = self.encode(x, train=True)
        if not self.block_grad:
            h = [h[-1]]
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss = self.loss(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, h
        else:
            return loss
        
    def forward_analytical_b(self, x, return_h=False):
        h = self.encode(x, train=True)
        if not self.block_grad:
            h = [h[-1]]
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss, optim_loss = self.loss.forward_analytical_b(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, optim_loss, h
        else:
            return loss, optim_loss
        
    def freeze_layer(self, layer_ind, loss_layer_ind=[]):
        for i in layer_ind:
            module = self.encoder.blocks[i]
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        for i in loss_layer_ind:
            self.loss.Wproj[i].weight.requires_grad = False
            self.loss.Wproj_asym[i].weight.requires_grad = False
        
        print('Frozen Parameters {}'.format([name for name, param in self.encoder.named_parameters() if not param.requires_grad]))
    
    def encode(self, x, train=False):
        if self.patch_input:
            x = ( # b, c, y, x
                x.unfold(2, self.patch_size, self.patch_size) # b, c, n_patches_y, x, patch_size
                .unfold(3, self.patch_size, self.patch_size) # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                .permute(2, 3, 0, 1, 4, 5) # n_patches_y, n_patches_x, b, c, patch_size, patch_size
            )
            patched_shape = x.shape
            x = x.flatten(end_dim=2)
        x = self.flatten(x)
        h = self.encoder(x)
        if self.patch_input and not train:
            h = [h_.reshape(patched_shape[0], patched_shape[1], patched_shape[2], -1).mean(dim=(0,1)) for h_ in h]
        return h
    
    def compute_proj_align(self):
        B_L = self.loss.Wproj[-1].weight.T
        cos_sim = {}
        for l, W_l in enumerate(self.loss.Wproj):
            W_l = self.loss.get_eff_weights(l).T#W_l.weight
            key = 'layer_{}'.format(l)
            W_ref = torch.eye(W_l.shape[0])
            for j in range(l+1, len(self.encoder.blocks)):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight.T
            W_ref = W_ref @ B_L
            for j in range(len(self.encoder.blocks)-1, self.fb_idx[l], -1):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight

            sim_score = torch.dot(W_l.flatten(), W_ref.flatten())/(torch.norm(W_l.flatten()+1e-10) * torch.norm(W_ref.flatten()+1e-10))
            cos_sim[key] = sim_score.item()
        return cos_sim
    
    def assign_proj_align(self):
        B_L = self.loss.Wproj[-1].weight.T
        for l, W_l in enumerate(self.loss.Wproj):
            key = 'layer_{}'.format(l)
            W_ref = torch.eye(W_l.weight.shape[0])
            for j in range(l+1, len(self.encoder.blocks)):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight.T
            W_ref = W_ref @ B_L
            for j in range(len(self.encoder.blocks)-1, self.fb_idx[l], -1):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight

            W_l.weight.data = W_ref.data.T

        B_L = self.loss.Wproj_asym[-1].weight.T
        for l, W_l in enumerate(self.loss.Wproj_asym):
            key = 'layer_{}'.format(l)
            W_ref = torch.eye(W_l.weight.shape[0])
            for j in range(l+1, len(self.encoder.blocks)):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight.T
            W_ref = W_ref @ B_L
            for j in range(len(self.encoder.blocks)-1, self.fb_idx[l], -1):
                W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight

            W_l.weight.data = W_ref.data.T


    def analyze_sim(self, x):
        self.eval()
        with torch.no_grad():
            h = self.encode(x, train=True)
            if not self.block_grad:
                h = [h[-1]]
            loss, sim_list = self.loss(h, return_sim=True)
        sim_dict = {}
        for l, sim in enumerate(sim_list):
            key = 'layer_{}'.format(l)
            sim_dict[key] = sim

        return loss, sim_dict
    
    def unfreeze_encoder(self):
        for module in self.encoder.blocks:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
    
    def freeze_encoder(self):
        for module in self.encoder.blocks:
            for param in module.parameters():
                param.requires_grad = False
        
    def compute_lw_error_grad(self, x, y, noise_layer=None):
        assert self.block_grad
        assert self.encoder.block_grad
        self.unfreeze_encoder()
        h = self.encode(x, train=True)
        for h_ in h:
            if h_.is_leaf:
                h_.requires_grad = True 
        if self.random_label is None:
            label_len = int(len(h[-1])) if self.loss.detach_c else int(len(h[-1])/2)
            self.random_label = torch.rand(label_len, device=self.device) < self.loss.transition_prob
        loss = self.loss(h, fixed_label = self.random_label, check_pos_only=False)
        
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.blocks, h)):
            key = 'layer_{}'.format(l)
            loss[l].backward(retain_graph = True)
            all_errors[key] = torch.autograd.grad(loss[l], x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            #all_bias_grads[key] = block[self.preact_index].bias.grad

        self.freeze_encoder()
        return all_errors, all_weight_grads, all_bias_grads
    

    def compute_ete_error_grad(self, x, y, noise_layer=None):
        self.encoder.block_grad = False
        x.requires_grad = True
        self.unfreeze_encoder()
        h = self.encode(x, train=True)
        z = h if self.block_grad else [h[-1]]
        if self.random_label is None:
            raise ValueError('need proper random label for comparison')
        loss = self.loss(z, fixed_label = self.random_label, check_pos_only=False)[-1]
        loss.backward(retain_graph = True)
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.blocks, h)):
            key = 'layer_{}'.format(l)
            all_errors[key] = torch.autograd.grad(loss, x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            #all_bias_grads[key] = block[self.preact_index].bias.grad
        self.random_label = None
        self.encoder.block_grad = True
        self.freeze_encoder()
        return all_errors, all_weight_grads, all_bias_grads


class ClappCNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = CNN(hparams, pooling=False, act_fun=nn.Identity if hparams['linear_nn'] else nn.ReLU, use_bias=False)
        self.custom_init = hparams['custom_init']
        if self.custom_init is not None:
            self.init_cnn()
        self.block_grad = hparams['layerwise']
        self.layer_dim = hparams['layer_dim'] if self.block_grad else [hparams['layer_dim'][-1]]
        self.device = hparams['device']

        self.custom_pool = hparams['custom_pool'] if ('custom_pool' in hparams) else None
        self.loss2d = False
        if self.custom_pool is not None:
            self.pool_modules =  nn.ModuleList(nn.AvgPool2d(kernel_size=int(w), stride=int(w), padding = 0) for w in self.custom_pool)
        dims = self.compute_dims(hparams['crop_size'])
        self.loss = ClappFlexLoss(hparams, layer_dimensions=dims if hparams['layerwise'] else [dims[-1]])
        self.flatten = nn.Flatten()
        self.random_label = None
        self.preact_index = 0
        self.fb_idx = hparams['fb_idx']
        

    def compute_dims(self,crop_size):
        x = torch.rand(1, 1, crop_size, crop_size)
        h = self.encoder(x)
        if self.custom_pool is not None:
            dims = [h_.shape[1]* int(h_.shape[2]//w) * int(h_.shape[3]//w) for h_, w in zip(h, self.custom_pool)]
        else:
            dims = [h_.shape[1] for h_ in h]
        return dims

    def init_cnn(self):

        if self.custom_init == 'default':
            return
        elif self.custom_init == 'orthogonal':
            for l_seq in self.encoder.layers:
                custom_init(l_seq[0], self.custom_init)
                print('{} weight initialized'.format(self.custom_init))
            return
        else:
            raise NotImplementedError
            # for l_seq in self.encoder.blocks:
            #     custom_init(l_seq[0], self.custom_init)
            #     print('{} weight initialized'.format(self.custom_init))
            # return

    def forward(self, x, return_h=False):
        h = self.encode(x, train=True)
        if not self.block_grad:
            h = [h[-1]]
       
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss = self.loss(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, h
        else:
            return loss

    def forward_analytical_b(self, x, return_h=False):
        h = self.encode(x, train=True)
        if not self.block_grad:
            h = [h[-1]]
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss, optim_loss = self.loss.forward_analytical_b(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, optim_loss, h
        else:
            return loss, optim_loss
        
    def freeze_layer(self, layer_ind, loss_layer_ind=[]):
        for i in layer_ind:
            module = self.encoder.layers[i]
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

        for i in loss_layer_ind:
            self.loss.Wproj[i].weight.requires_grad = False
            self.loss.Wproj_asym[i].weight.requires_grad = False
        
        print('Frozen Parameters {}'.format([name for name, param in self.encoder.named_parameters() if not param.requires_grad]))
    
    def process_h(self, h, train=False):
        if not train:
            if self.custom_pool is not None:
                h = [self.pool_modules[i](h_).flatten(start_dim=1) for i, h_ in enumerate(h)]
            else:
                h = [torch.mean(h_, dim=(2, 3)) for h_ in h]
        elif not self.loss2d:
            if self.custom_pool is not None:
                h = [self.pool_modules[i](h_).flatten(start_dim=1) for i, h_ in enumerate(h)]
            else:
                h = [torch.mean(h_, dim=(2, 3)) for h_ in h]
        return h
    
    def encode(self, x, train=False, return_original=False):
        h = self.encoder(x)

        if return_original:
            return h
        else:
            return self.process_h(h, train=train)

    
    def compute_proj_align(self):
        B_L = self.loss.Wproj[-1].weight
        cos_sim = {}
        for l, W_l in enumerate(self.loss.Wproj):
            key = 'layer_{}'.format(l)
            # W_l = self.loss.get_eff_weights(l)#W_l.weight
            # W_ref = torch.eye(W_l.shape[0])
            # for j in range(l+1, len(self.encoder.layers)):
            #     W_ref = W_ref @ self.encoder.layers[j][self.preact_index].weight.T
            # W_ref = W_ref @ B_L
            # for j in range(len(self.encoder.layers)-1, self.fb_idx[l], -1):
            #     W_ref = W_ref @ self.encoder.layers[j][self.preact_index].weight

            # sim_score = torch.dot(W_l.flatten(), W_ref.flatten())/(torch.norm(W_l.flatten()+1e-10) * torch.norm(W_ref.flatten()+1e-10))
            cos_sim[key] = 0 #sim_score.item()
        return cos_sim
                
    def analyze_sim(self, x):
        self.eval()
        with torch.no_grad():
            h = self.encode(x, train=True)
            if not self.block_grad:
                h = [h[-1]]
            loss, sim_list = self.loss(h, return_sim=True)
        sim_dict = {}
        for l, sim in enumerate(sim_list):
            key = 'layer_{}'.format(l)
            sim_dict[key] = sim

        return loss, sim_dict
    
    def unfreeze_encoder(self):
        for module in self.encoder.layers:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
    
    def freeze_encoder(self):
        for module in self.encoder.layers:
            for param in module.parameters():
                param.requires_grad = False
        
    def compute_lw_error_grad(self, x, y, noise_layer=None):
        assert self.block_grad
        assert self.encoder.block_grad
        self.unfreeze_encoder()
        h = self.encode(x, train=True, return_original=True)
        for h_ in h:
            if h_.is_leaf:
                h_.requires_grad = True 
        if self.random_label is None:
            label_len = int(len(h[-1])) if self.loss.detach_c else int(len(h[-1])/2)
            self.random_label = torch.rand(label_len, device=self.device) < self.loss.transition_prob
        loss = self.loss(self.process_h(h, train=True), fixed_label = self.random_label, check_pos_only=False)
        
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.layers, h)):
            key = 'layer_{}'.format(l)
            loss[l].backward(retain_graph = True)
            all_errors[key] = torch.autograd.grad(loss[l], x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            #all_bias_grads[key] = block[self.preact_index].bias.grad

        self.freeze_encoder()
        return all_errors, all_weight_grads, all_bias_grads
    

    def compute_ete_error_grad(self, x, y, noise_layer=None):
        self.encoder.block_grad = False
        x.requires_grad = True
        self.unfreeze_encoder()
        h = self.encode(x, train=True, return_original=True)
        z = self.process_h(h, train=True) if self.block_grad else [self.process_h(h, train=True)[-1]]
        if self.random_label is None:
            raise ValueError('need proper random label for comparison')
        loss = self.loss(z, fixed_label = self.random_label, check_pos_only=False)[-1]
        loss.backward(retain_graph = True)
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.layers, h)):
            key = 'layer_{}'.format(l)
            all_errors[key] = torch.autograd.grad(loss, x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            #all_bias_grads[key] = block[self.preact_index].bias.grad
        self.random_label = None
        self.encoder.block_grad = True
        self.freeze_encoder()
        return all_errors, all_weight_grads, all_bias_grads

def compute_similarity(model, testloader, device, inquired_layer='layer_0'):
    model.eval()
    similairty = {}
    for x, y in tqdm(testloader):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        loss, sim_dict = model.analyze_sim(x)
        merge_in_dict(similairty, sim_dict)
    if inquired_layer == 'all':
        result_dict = {}
        for key in similairty.keys():
            cocat_sim = torch.concat(similairty[key])
            result_dict[key+'_sim'] = cocat_sim.mean().item()
            result_dict[key+'_sim_std'] = cocat_sim.std().item()
            if key == 'layer_0':
                print('layer 0 similarity: mean {}, std {}'.format(result_dict[key+'_sim'], result_dict[key+'_sim_std']))
        return result_dict
    else:
        if inquired_layer not in similairty:
            raise ValueError('inquired layer {} not found in model'.format(inquired_layer))
        cocat_sim = torch.concat(similairty[inquired_layer])
        sim_mean = cocat_sim.mean().item()
        sim_std = cocat_sim.std().item()
        print('layer {} similarity: mean {}, std {}'.format(inquired_layer, sim_mean, sim_std))
        return sim_mean, sim_std

def compare_fb_grad(model, testloader, device, inquired_layer='layer_0', noise_layer=None):
    model.eval()

    dfa_errors = {}
    dfa_weight_grads = {}
    dfa_bias_grads = {}
    auto_errors = {}
    auto_weight_grads = {}
    auto_bias_grads = {}
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        dfa_error, dfa_weight_grad, dfa_bias_grad = model.compute_lw_error_grad(x, y, noise_layer = noise_layer)
        merge_in_dict(dfa_errors, dfa_error)
        merge_in_dict(dfa_weight_grads, dfa_weight_grad)
        merge_in_dict(dfa_bias_grads, dfa_bias_grad)
        model.zero_grad()
        auto_error, auto_weight_grad, auto_bias_grad = model.compute_ete_error_grad(x, y, noise_layer = noise_layer)
        merge_in_dict(auto_errors, auto_error)
        merge_in_dict(auto_weight_grads, auto_weight_grad)
        merge_in_dict(auto_bias_grads, auto_bias_grad)

    if inquired_layer == 'all':
        result_dict = {}
        for key in dfa_errors.keys():

            e_mean, e_std = compute_cos_sim(dfa_errors[key], auto_errors[key])
            w_mean, w_std = compute_cos_sim(dfa_weight_grads[key], auto_weight_grads[key])
            #b_mean, b_std = compute_cos_sim(bias_grads[key], auto_bias_grads[key])
            result_dict[key+'_error'] = e_mean
            result_dict[key+'_error_std'] = e_std
            result_dict[key+'_weight'] = w_mean
            result_dict[key+'_weight_std'] = w_std
            #result_dict[key+'_bias'] = (b_mean, b_std)
            if key == 'layer_0':
                print('layer 0 error approximation {}, weight approximation {}'.format(e_mean, w_mean))

        return result_dict
    else:
        e_mean, e_std = compute_cos_sim(dfa_errors[inquired_layer], auto_errors[inquired_layer])
        w_mean, w_std = compute_cos_sim(dfa_weight_grads[inquired_layer], auto_weight_grads[inquired_layer])
        b_mean, b_std = compute_cos_sim(dfa_bias_grads[inquired_layer], auto_bias_grads[inquired_layer])
        print('layer {} error approximation: mean {}, std {}'.format(inquired_layer, e_mean, e_std))
        return e_mean, w_mean, b_mean