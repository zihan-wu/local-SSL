import torch
import numpy as np
import torch.nn as nn
from encoders import MLP
from loss import ClappFlexLoss
from utils import merge_in_dict, compute_cos_sim
from tqdm import tqdm

def compute_activation_derivative(x, act_fun):
    if isinstance(act_fun, nn.ReLU):
        return (x>0).type(torch.float)
    elif isinstance(act_fun, nn.Sigmoid):
        act_x = nn.functional.sigmoid(x)
        return act_x * (1-act_x)
    elif isinstance(act_fun, nn.Identity):
        return torch.ones_like(x)
    else:
        raise NotImplementedError('Derivative for current activation {} is not computed'.format(act_fun))


class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, out_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.layer(x)
        loss = self.loss(logits, y)
        return loss, logits
    


class CLAPP_FB(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.patch_input = hparams['patch_input']
        if self.patch_input:
            hparams['input_dim'] = hparams['patch_size'] ** 2
            self.patch_size = hparams['patch_size']
        self.encoder = MLP(hparams)
        self.block_grad = hparams['layerwise']
        self.layer_dim = hparams['layer_dim'] if self.block_grad else [hparams['layer_dim'][-1]]
        self.flatten = nn.Flatten()
        self.block_grad = hparams['layerwise']
        self.patch_input = hparams['patch_input']
        self.device = hparams['device']
        self.preact_index = self.encoder.preact_ind
        #self.final_loss = HingeLoss(hparams)
        
        self.fb_idx = hparams['fb_idx']
        self.ete_with_layer_loss = hparams['ete_with_layer_loss']
        self.fb_loss = ClappFlexLoss(hparams, layer_dimensions=hparams['layer_dim'] if self.ete_with_layer_loss else None) #ClappFBLoss(hparams)
        self.random_label = None
        self.freeze_fb = hparams.get('freeze_fb', False)
        if self.freeze_fb:
            for p in self.fb_loss.Wproj[:-1].parameters():
                p.requires_grad = False
            if hparams['asym_fb']:
                for p in self.fb_loss.Wproj_asym[:-1].parameters():
                    p.requires_grad = False

    def process_h(self, h):
        if (not self.block_grad) and (not self.ete_with_layer_loss):
            h = [h[-1]]
        elif self.ete_with_layer_loss:
            h = [h_.detach() for h_ in h[:-1]] + [h[-1]]
        return h
    
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
    
    def forward(self, x, return_h=False):
        h = self.encode(x, train=True)
        h = self.process_h(h)
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss = self.fb_loss(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, h
        else:
            return loss
    
    def forward_fb_with_grad(self, x):
        if self.freeze_fb:
            return 0
        
        for params in self.fb_loss.parameters():
            params.grad = None
        self.encoder.block_grad = False
        h = self.encode(x, train=True)
        h = self.process_h(h)
        loss = self.fb_loss.forward_best_grad_pred(h)
        self.encoder.block_grad = self.block_grad
        return loss

    def scale_pred_grad(self, scale_):
        for proj in self.fb_loss.Wproj:
            proj.weight.grad = proj.weight.grad * scale_


    def forward_analytical_b(self, x, return_h=False):
        h = self.encode(x, train=True)
        h = self.process_h(h)
        #print('dim {}'.format([h_.shape for h_ in h]))
        loss, optim_loss = self.fb_loss.forward_analytical_b(h)
        #loss, eq_grads, all_projs, all_labels = self.loss.compute_ep_grad(h, all_projs=None, all_labels=None, c_list=h)
        if return_h:
            return loss, optim_loss, h
        else:
            return loss, optim_loss

    def compute_proj_align(self):
        B_L = self.fb_loss.Wproj[-1].weight.T
        cos_sim = {}
        for l, W_l in enumerate(self.fb_loss.Wproj):
            W_l = self.fb_loss.get_eff_weights(l).T#W_l.weight
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
    
    def compute_mask_approx(self, x, y, noise_layer=None, method='per_sample'):

        if self.patch_input:
            x = ( # b, c, y, x
                x.unfold(2, self.patch_size, self.patch_size) # b, c, n_patches_y, x, patch_size
                .unfold(3, self.patch_size, self.patch_size) # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                .permute(2, 3, 0, 1, 4, 5) # n_patches_y, n_patches_x, b, c, patch_size, patch_size
            )
            patched_shape = x.shape
            x = x.flatten(end_dim=2)

        x = self.flatten(x)
        h_list, preact_list = self.encoder.forward_with_preact(x)
        loss = self.fb_loss(h_list[1:] if self.block_grad else [h_list[-1]])
        preact_derivative_list = [compute_activation_derivative(preact, block[self.preact_index + 1]) for block, preact in zip(self.encoder.blocks, preact_list)]

        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        delta = torch.autograd.grad(loss[-1], h_list[-1])[0].detach()

        for l in range(len(self.encoder.blocks)-1, -1, -1):
            W_l = self.encoder.blocks[l][self.preact_index].weight
            key = 'layer_{}'.format(l)

            #sim_score = torch.dot(W_l.flatten(), W_ref.flatten())/(torch.norm(W_l.flatten()+1e-10) * torch.norm(W_ref.flatten()+1e-10))
            all_errors[key] = delta
            if method == 'per_sample':
                delta = preact_derivative_list[l] * delta
            elif method == 'per_batch':
                delta = preact_derivative_list[l].mean(dim=0) * delta
            elif method == 'no_mask':
                delta = delta
            else:
                raise NotImplementedError('method {} not implemented'.format(method))

            all_weight_grads[key] = delta.T @ h_list[l]
            all_bias_grads[key] = delta.sum(dim=0)

            delta = delta @ W_l
            
            
            # for j in range(len(self.encoder.blocks)-1, self.fb_idx[l], -1):
            #     W_ref = W_ref @ self.encoder.blocks[j][self.preact_index].weight

        return all_errors, all_weight_grads, all_bias_grads

    def compute_error_grad(self, x, y, noise_layer=None):

        h = self.encode(x, train=True)
        h = self.process_h(h)
        #print('dim {}'.format([h_.shape for h_ in h]))
        if self.random_label is None:
            label_len = int(len(h[-1])) if self.fb_loss.detach_c else int(len(h[-1])/2)
            self.random_label = torch.rand(label_len, device=self.device) < self.fb_loss.transition_prob
        loss = self.fb_loss(h, fixed_label = self.random_label)
        loss.sum().backward(retain_graph = True)
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.blocks, h)):
            key = 'layer_{}'.format(l)
            all_errors[key] = torch.autograd.grad(loss[l], x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            all_bias_grads[key] = block[self.preact_index].bias.grad
        return all_errors, all_weight_grads, all_bias_grads

    def compute_auto_error_grad(self, x, y, noise_layer=None):

        self.encoder.block_grad = False
        h = self.encode(x, train=True)
        h = self.process_h(h)
        #print('dim {}'.format([h_.shape for h_ in h]))
        if self.random_label is None:
            label_len = int(len(h[-1])) if self.fb_loss.detach_c else int(len(h[-1])/2)
            self.random_label = torch.rand(label_len, device=self.device) < self.fb_loss.transition_prob
        loss = self.fb_loss(h, fixed_label = self.random_label)
        loss[-1].backward(retain_graph = True)
        all_errors = {}
        all_weight_grads = {}
        all_bias_grads = {}
        for l, (block, x) in enumerate(zip(self.encoder.blocks, h)):
            key = 'layer_{}'.format(l)
            all_errors[key] = torch.autograd.grad(loss[-1], x, retain_graph=True)[0]
            all_weight_grads[key] = block[self.preact_index].weight.grad
            all_bias_grads[key] = block[self.preact_index].bias.grad
        
        self.encoder.block_grad = self.block_grad
        self.random_label = None
        return all_errors, all_weight_grads, all_bias_grads
    

def compare_fb_grad(model, testloader, device, inquired_layer='layer_0', noise_layer=None, analyze_mask=None):

    dfa_errors = {}
    dfa_weight_grads = {}
    dfa_bias_grads = {}
    auto_errors = {}
    auto_weight_grads = {}
    auto_bias_grads = {}
    if analyze_mask is not None:
        mask_errors = {}
        mask_weight_grads = {}
        mask_bias_grads = {}
    for x, y in tqdm(testloader):
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        dfa_error, dfa_weight_grad, dfa_bias_grad = model.compute_error_grad(x, y, noise_layer = noise_layer)
        merge_in_dict(dfa_errors, dfa_error)
        merge_in_dict(dfa_weight_grads, dfa_weight_grad)
        merge_in_dict(dfa_bias_grads, dfa_bias_grad)
        model.zero_grad()
        auto_error, auto_weight_grad, auto_bias_grad = model.compute_auto_error_grad(x, y, noise_layer = noise_layer)
        merge_in_dict(auto_errors, auto_error)
        merge_in_dict(auto_weight_grads, auto_weight_grad)
        merge_in_dict(auto_bias_grads, auto_bias_grad)
        if analyze_mask is not None:
            mask_error, mask_weight_grad, mask_bias_grad = model.compute_mask_approx(x, y, noise_layer = noise_layer, method=analyze_mask)
            merge_in_dict(mask_errors, mask_error)
            merge_in_dict(mask_weight_grads, mask_weight_grad)
            merge_in_dict(mask_bias_grads, mask_bias_grad)


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
            if analyze_mask is not None:
                me_mean, me_std = compute_cos_sim(dfa_errors[key], mask_errors[key])
                we_mean, we_std = compute_cos_sim(dfa_weight_grads[key], mask_weight_grads[key])
                result_dict[key+'_mask_error'] = me_mean
                result_dict[key+'_mask_error_std'] = me_std
                result_dict[key+'_mask_weight'] = we_mean
                result_dict[key+'_mask_weight_std'] = we_std
                if key == 'layer_0':
                    print('layer 0 mask error approximation {}, weight approximation {}'.format(me_mean, we_mean))

        return result_dict
    else:
        e_mean, e_std = compute_cos_sim(dfa_errors[inquired_layer], auto_errors[inquired_layer])
        w_mean, w_std = compute_cos_sim(dfa_weight_grads[inquired_layer], auto_weight_grads[inquired_layer])
        b_mean, b_std = compute_cos_sim(dfa_bias_grads[inquired_layer], auto_bias_grads[inquired_layer])
        print('layer {} error approximation: mean {}, std {}'.format(inquired_layer, e_mean, e_std))
        return e_mean, w_mean, b_mean
