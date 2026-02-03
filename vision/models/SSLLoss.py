import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import numpy as np
from numpy.random import choice
import os

from vision.utils import model_utils


class CPLoss(nn.Module): # Contrastive Predictive
    def __init__(self, opt, h_dims, proj_kernel=1, proj_stride=1, save_vars=False, diff_layer_pred=False, fb_idx=None): # in_channels: z, out_channels: c
        super().__init__()
        self.opt = opt
        self.h_dims = [h_dims[-1]] if opt.ete_training else h_dims
        self.negative_samples = self.opt.negative_samples
        self.avoid_same_neg_sample = self.opt.avoid_same_neg_sample
        self.diff_layer_pred = diff_layer_pred
        self.contrast_mode = self.opt.contrast_mode
        self.detach_c = self.opt.detach_c
        self.either_pos_or_neg_update = self.opt.either_pos_or_neg_update
        self.which_update = 'both'
        self.spatial_z = False
        
        if fb_idx is not None:
            self.fb_idx = fb_idx
        elif opt.customize_fb_idx is not None:
            self.fb_idx = [int(i)-1 for i in opt.customize_fb_idx.split('-')]
        else:
            self.fb_idx = range(opt.model_splits)
        
        if opt.customize_loss_pool is not None:
            if opt.adaptive_loss_pool:
                self.pool_module = nn.ModuleList(nn.AdaptiveAvgPool2d(output_size=int(w)) for w in opt.customize_loss_pool.split('-'))
            else:
                self.pool_module = nn.ModuleList(nn.AvgPool2d(kernel_size=int(w), stride=int(w), padding = 0) for w in opt.customize_loss_pool.split('-'))
        else:
            self.pool_module = None
        self.init_proj(proj_kernel, proj_stride)
        
    def init_proj(self, proj_kernel, proj_stride):

        if self.opt.use_transpose_pred and kernel_ > 1:
            self.spatial_z = True
            if self.opt.predict_module_num in ['-1', 'fb', 'fb_only']:
                kernel_ = proj_kernel
                stride_ = proj_stride
                padding_ = (kernel_-1)//2
                
            else:
                raise NotImplementedError('have not encounter the scenario for transpose2dConv')
            
        if self.opt.customize_loss_pool is not None:
            if self.opt.adaptive_loss_pool:
                self.h_dims = [dims[1] * int(w) * int(w) for w, dims in zip(self.opt.customize_loss_pool.split('-'), self.h_dims)]
            else:
                self.h_dims = [dims[1] * int(dims[2]/int(w)) * int(dims[3]/int(w)) for w, dims in zip(self.opt.customize_loss_pool.split('-'), self.h_dims)]
        else:
            self.h_dims = [dims[1] for dims in self.h_dims]
        
        in_channels = self.h_dims if self.opt.ete_training else [self.h_dims[i] for i in self.fb_idx]
        if self.opt.predict_module_num == 'both':
            in_channels = [self.h_dims[i] + self.h_dims[id] for i, id in enumerate(self.fb_idx)]
        out_channels = self.h_dims
        
        # kernel_ = 1
        # stride_ = 1
        # padding_ = 1
        # self.W_k = nn.ModuleList(
        #             nn.Conv2d(c_in, c_out, kernel_size=kernel_, stride=stride_, padding=padding_, bias=False) # in_channels: z, out_channels: c
        #             for c_in, c_out in zip(in_channels, out_channels)
        #         )
        if self.opt.identity_projection:
            self.W_k = nn.ModuleList(
                        nn.Identity() # in_channels: z, out_channels: c
                        for c_in, c_out in zip(in_channels, out_channels)
                    )
        elif self.opt.use_proj_head:
            self.W_k = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(c_in, self.opt.low_rank_dim, bias=False),
                    nn.Linear(self.opt.low_rank_dim, c_out, bias=False)) # in_channels: z, out_channels: c
                for c_in, c_out in zip(in_channels, out_channels)
            )
        else:
            self.W_k = nn.ModuleList(
                        nn.Linear(c_in, c_out, bias=False) # in_channels: z, out_channels: c
                        for c_in, c_out in zip(in_channels, out_channels)
                    )
        self.n_loss = len(self.W_k)

        if self.opt.freeze_W_pred: #or (self.opt.dfa and diff_layer_pred): # freeze prediction weights W_k
            print('Warning! W_pred is frozen in this loss')
            if self.opt.unfreeze_last_W_pred:
                params_to_freeze = self.W_k[:-1].parameters()
            else:
                params_to_freeze = self.W_k.parameters()
            for p in params_to_freeze:
                p.requires_grad = False
    
    def get_params(self, idx):
        return self.W_k[idx].parameters()
    
    def compute_norm(self, idx):
        if self.opt.use_proj_head:
            raise NotImplementedError
        else:
            return torch.square(self.W_k[idx].weight).sum()
             

    def sample_negatives(self, z, rand_index, cur_device):
        if self.opt.determined_neg_samples:
            return self.sample_next_negatives(z, rand_index, cur_device)
        if (rand_index is None) or (not self.opt.unified_random_sampling):
            rand_index = torch.randint(z.shape[0], # upper limit: b
                (z.shape[0] * self.negative_samples,), # shape: b, assumes n=1 neg. samples, 
                dtype=torch.long,
                device=cur_device,
            )

        #z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1], z.shape[2], z.shape[3])) # n, B, C, H, W
        z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1])) # n, B, C, H, W

        return z_neg, rand_index
    
    def sample_next_negatives(self, z, rand_index, cur_device):
        assert self.negative_samples == 1, "only support 1 negative sample for next sample strategy"

        current_id = torch.arange(z.shape[0], device=cur_device)
        next_id = (current_id + 1) % z.shape[0]

        #z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1], z.shape[2], z.shape[3])) # n, B, C, H, W
        z_neg = z[next_id].reshape((1, z.shape[0], z.shape[1])) # n, B, C, H, W

        return z_neg, rand_index
    
    def process_reps(self, h_list):
        if self.pool_module is not None:
            new_h = [self.pool_module[i](h).flatten(start_dim=1) for i, h in enumerate(h_list)]
            #print('shape {}'.format([h_.shape for h_ in new_h]))
        else:
            new_h = [torch.mean(h, (2,3)) if h.dim()>2 else h for i, h in enumerate(h_list)]
        return new_h
    

    def update_proj(self, W, loss, z_pos, z_neg, c):
        if self.opt.retain_proj_grads:
            grad = torch.autograd.grad(loss.mean(), W.weight, retain_graph=True, create_graph=True)[0]
            wc = c @ ((W.weight - self.opt.learning_rate * grad).T) # B, C_in (* H * W)

        else:
            grad = torch.autograd.grad(loss.mean(), W.weight, retain_graph=True)[0]
            W.weight.data = W.weight.data.detach() - self.opt.learning_rate * grad.detach()
            if self.opt.normalize_pred:
                self.normalize_proj(W)  

            wc = W(c) #B, C_in (* H * W)

        u_pos = (z_pos * wc) # B, C_out (* H * W)
        u_neg = (z_neg * wc) # n, B, C_out (* H * W)

        return u_pos, u_neg
    
    def normalize_proj(self, W):
        if self.opt.use_proj_head:
            raise NotImplementedError
        else:
            W.weight.data = F.normalize(W.weight.data, dim=1)
    
    def create_context(self, h_list, batch_size, i):
        if self.opt.asymmetric_W_pred:
            if self.opt.predict_module_num == 'both':
                return torch.cat([h_list[self.fb_idx[i]].detach(), h_list[i].detach()], dim=1)
            else:
                return h_list[self.fb_idx[i]].detach()
            
        else:
            if self.opt.predict_module_num == 'both':
                return torch.cat([h_list[self.fb_idx[i]][batch_size:], h_list[i][batch_size:]], dim=1)
            else:
                return h_list[self.fb_idx[i]][batch_size:]

    def forward(self, h_list, rand_index=None, rand_fixation=None, idx_range=None, proj_c = False):
        if idx_range is None:
            idx_range = list(range(self.n_loss))
        h_list = self.process_reps(h_list)
        cur_device = h_list[0].get_device() if self.opt.device.type != "cpu" else self.opt.device
        accuracies = torch.zeros(1, self.n_loss, device=cur_device)
        loss_dict = {'loss': torch.zeros(1, self.n_loss, device=cur_device)}
        if self.opt.log_pos_neg:
            loss_dict['loss_pos'] = torch.zeros(1, self.n_loss, device=cur_device)
            loss_dict['loss_neg'] = torch.zeros(1, self.n_loss, device=cur_device)
        for i, (h, W) in enumerate(zip(h_list, self.W_k)):
            if self.opt.normalize_pred:
                self.normalize_proj(W)
            if i not in idx_range:
                continue
            batch_size = len(h)//2
            # sample negative data
            z_pos = torch.vstack([h[batch_size:], h[:batch_size]]) if self.opt.asymmetric_W_pred else h[:batch_size] # B, C_out(* H * W)
            z_neg, rand_index = self.sample_negatives(z_pos, rand_index, cur_device) # n, B, C_out(* H * W)

            # project contextual representation
            #c = h_list[self.fb_idx[i]].detach() if self.opt.asymmetric_W_pred else h_list[self.fb_idx[i]][batch_size:]
            c = self.create_context(h_list, batch_size, i)
            wc = W(c) #B, C_in (* H * W)

            # compute score and loss, default averages the spatial neurons
            u_pos = (z_pos * wc) # B, C_out (* H * W)
            u_neg = (z_neg * wc) # n, B, C_out (* H * W)

            loss, rand_fixation = self.compute_loss(u_pos.sum(dim=-1), u_neg.sum(dim=-1), rand_fixation, cur_device, i)

            for _ in range(self.opt.update_proj_steps):
                u_pos, u_neg = self.update_proj(W, loss['loss'], z_pos, z_neg, c)
                loss, rand_fixation = self.compute_loss(u_pos.sum(dim=-1), u_neg.sum(dim=-1), rand_fixation, cur_device, i)

            
            for key, value in loss_dict.items():
                value[:, i] = loss[key]

        return loss_dict, rand_index, rand_fixation, accuracies
    
    def forward_best_grad_pred(self, raw_h_list, rand_index=None, rand_fixation=None, idx_range=None, proj_c = False):
        if idx_range is None:
            idx_range = list(range(self.n_loss))
        h_list = self.process_reps(raw_h_list)
        cur_device = h_list[0].get_device() if self.opt.device.type != "cpu" else self.opt.device
        accuracies = torch.zeros(1, self.n_loss, device=cur_device)
        loss_dict = {'loss': torch.zeros(1, self.n_loss, device=cur_device)}
        if self.opt.log_pos_neg:
            loss_dict['loss_pos'] = torch.zeros(1, self.n_loss, device=cur_device)
            loss_dict['loss_neg'] = torch.zeros(1, self.n_loss, device=cur_device)
        for i in range(self.n_loss-1, -1, -1):
            h = h_list[i]
            W = self.W_k[i]
            raw_h = raw_h_list[i]
            if i < self.n_loss - 1:
                #print('loss {} for layer {}'.format(loss_dict['loss'][0, -1], i))
                ideal_grad_pred = torch.autograd.grad(loss_dict['loss'][0, -1], raw_h, retain_graph=True)[0]
                # h = h.detach()
                # raw_h.requires_grad = True

            if self.opt.normalize_pred:
                self.normalize_proj(W)
            if i not in idx_range:
                continue
            batch_size = len(h)//2
            # sample negative data
            z_pos = torch.vstack([h[batch_size:], h[:batch_size]]) if self.opt.asymmetric_W_pred else h[:batch_size] # B, C_out(* H * W)
            z_neg, rand_index = self.sample_negatives(z_pos, rand_index, cur_device) # n, B, C_out(* H * W)

            # project contextual representation
            #c = h_list[self.fb_idx[i]].detach() if self.opt.asymmetric_W_pred else h_list[self.fb_idx[i]][batch_size:]
            c = self.create_context(h_list, batch_size, i)
            wc = W(c) #B, C_in (* H * W)

            # compute score and loss, default averages the spatial neurons
            u_pos = (z_pos * wc) # B, C_out (* H * W)
            u_neg = (z_neg * wc) # n, B, C_out (* H * W)

            loss, rand_fixation = self.compute_loss(u_pos.sum(dim=-1), u_neg.sum(dim=-1), rand_fixation, cur_device, i)
            
            for key, value in loss_dict.items():
                value[:, i] = loss[key]

            final_loss = loss_dict['loss']
            if i == self.n_loss - 1:
                #W.weight.grad = torch.autograd.grad(final_loss[0, i], W.weight, retain_graph=True)[0]
                continue
            else:
                loss_i = loss['loss']
                grad_pred = torch.autograd.grad(loss_i, raw_h, retain_graph=True, create_graph=True)[0]

                fb_loss_i = ((ideal_grad_pred - grad_pred)**2).sum() # #(1 - F.cosine_similarity(ideal_grad_pred, grad_pred, dim=-1)).sum() #((ideal_grad_pred - grad_pred)**2).sum()
                W.weight.grad = torch.autograd.grad(fb_loss_i, W.weight, retain_graph=True)[0]
                #print('Layer {} feedback loss {}, with weight grad norm {}'.format(i, fb_loss_i.item(), W.weight.grad.norm().item()))
                loss_dict['loss'][0, i] = fb_loss_i/(ideal_grad_pred **2).sum().detach() # normalize by ideal grad norm


        return loss_dict, rand_index, rand_fixation, accuracies

    def compute_loss(self, u_pos, u_neg, rand_fixation, cur_device, i):
        if self.contrast_mode=='hinge':
            loss_pos = F.relu(1 - u_pos)
            loss_neg = F.relu(1 + u_neg).mean(dim=0)

            if self.either_pos_or_neg_update:
                if (rand_fixation is None) or (not self.opt.unified_random_sampling):
                    rand_fixation = (torch.rand(len(u_pos), device=cur_device) > 0.5)
                loss = torch.where(rand_fixation, loss_pos, loss_neg)
            else:
                loss = self.opt.pos_coeff * loss_pos + loss_neg
        elif self.contrast_mode=='softhinge':
            loss_pos = -F.logsigmoid(u_pos - 1)
            loss_neg = -F.logsigmoid(-u_neg - 1).mean(dim=0)

            if self.either_pos_or_neg_update:
                if (rand_fixation is None) or (not self.opt.unified_random_sampling):
                    rand_fixation = (torch.rand(len(u_pos), device=cur_device) > 0.5)
                loss = torch.where(rand_fixation, loss_pos, loss_neg)
            else:
                loss = self.opt.pos_coeff * loss_pos + loss_neg
        elif self.contrast_mode=='phyll':
            loss = self.opt.phyll_theta*torch.nn.functional.softplus(u_neg.mean(dim=0) - u_pos, beta=self.opt.phyll_theta) #self.opt.phyll_theta*
            loss_neg = u_neg.mean(dim=0)
            loss_pos = u_pos 
        elif self.contrast_mode=='linear':
            loss_neg = u_neg.mean(dim=0)
            loss_pos = -u_pos 
            loss = u_neg.mean(dim=0) - u_pos
        else:
            raise NotImplementedError

        output = {'loss': loss.mean(), 'loss_pos': u_pos.mean(), 'loss_neg': u_neg.mean()}
        if self.opt.pred_decay > 0:
            output['loss'] = output['loss'] + self.opt.pred_decay * self.compute_norm(i)
        return output, rand_fixation





class CP2DLoss(nn.Module): # Contrastive Predictive
    def __init__(self, opt, h_dims, proj_kernel=1, proj_stride=1, save_vars=False, diff_layer_pred=False): # in_channels: z, out_channels: c
        super().__init__()
        self.opt = opt
        self.h_dims = h_dims
        self.negative_samples = self.opt.negative_samples
        self.avoid_same_neg_sample = self.opt.avoid_same_neg_sample
        self.diff_layer_pred = diff_layer_pred
        self.contrast_mode = self.opt.contrast_mode
        self.detach_c = self.opt.detach_c
        self.either_pos_or_neg_update = self.opt.either_pos_or_neg_update
        self.which_update = 'both'
        self.spatial_z = False
        

        if opt.customize_fb_idx is not None:
            self.fb_idx = [int(i)-1 for i in opt.customize_fb_idx.split('-')]
        else:
            self.fb_idx = range(opt.model_splits)
        
        # if opt.customize_loss_pool is not None:
        #     self.pool_module = nn.ModuleList(nn.AvgPool2d(kernel_size=int(w), stride=int(w), padding = 0) for w in opt.customize_loss_pool.split('-'))
        # else:
        #     self.pool_module = None
        self.init_proj(proj_kernel, proj_stride)
        
    def init_proj(self, proj_kernel, proj_stride):


        in_channels = [dims[1] for dims in self.h_dims]
        out_channels = [self.h_dims[-1][1]] * len(self.h_dims) if self.opt.use_proj_head else [in_channels[i] for i in self.fb_idx]
        
        self.W_k = nn.ModuleList(
                    nn.Conv2d(c_in, c_out, bias=False, kernel_size=self.opt.config['layer_configs'][i_l]['loss_kernel'], 
                              stride=self.opt.config['layer_configs'][i_l]['loss_stride'], padding=self.opt.config['layer_configs'][i_l]['loss_pad']) # in_channels: z, out_channels: c
                    for i_l, (c_in, c_out) in enumerate(zip(in_channels, out_channels))
                )
        if self.opt.use_asym_proj_head:
            self.W_k_mirror = nn.ModuleList(
                    nn.Conv2d(c_in, c_out, bias=False, kernel_size=self.opt.config['layer_configs'][i_l]['loss_kernel'], 
                              stride=self.opt.config['layer_configs'][i_l]['loss_stride'], padding=self.opt.config['layer_configs'][i_l]['loss_pad']) # in_channels: z, out_channels: c
                for i_l, (c_in, c_out) in enumerate(zip(in_channels, out_channels))
            )
        self.n_loss = len(self.W_k)
    
    def compute_norm(self, idx):
        if self.opt.use_proj_head:
            raise NotImplementedError
        else:
            return torch.square(self.W_k[idx].weight).sum()
    
    def get_params(self, idx):
        if self.opt.use_asym_proj_head:
            return list(self.W_k[idx].parameters()) +  list(self.W_k_mirror[idx].parameters())
        else:
            return self.W_k[idx].parameters() 

    def sample_negatives(self, z, rand_index, cur_device):
        if (rand_index is None) or (not self.opt.unified_random_sampling):
            rand_index = torch.randint(z.shape[0], # upper limit: b
                (z.shape[0] * self.negative_samples,), # shape: b, assumes n=1 neg. samples, 
                dtype=torch.long,
                device=cur_device,
            )

        z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1], z.shape[2], z.shape[3])) # n, B, C, H, W
        #z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1])) # n, B, C (* H * W)

        return z_neg, rand_index
    
    def process_reps(self, h_list):
        if self.pool_module is not None:
            new_h = [self.pool_module[i](h).flatten(start_dim=1) for i, h in enumerate(h_list)]
            #print('shape {}'.format([h_.shape for h_ in new_h]))
        else:
            new_h = [torch.mean(h, (2,3)) for i, h in enumerate(h_list)]
        return new_h

            
    def forward(self, h_list, rand_index=None, rand_fixation=None, idx_range=None, proj_c = False):
        if idx_range is None:
            idx_range = list(range(self.n_loss))
        #h_list = self.process_reps(h_list)
        cur_device = h_list[0].get_device() if self.opt.device.type != "cpu" else self.opt.device
        accuracies = torch.zeros(1, self.n_loss, device=cur_device)
        loss_dict = {'loss': torch.zeros(1, self.n_loss, device=cur_device)}
        if self.opt.log_pos_neg:
            loss_dict['loss_pos'] = torch.zeros(1, self.n_loss, device=cur_device)
            loss_dict['loss_neg'] = torch.zeros(1, self.n_loss, device=cur_device)
        for i, (h, W) in enumerate(zip(h_list, self.W_k)):
            if i not in idx_range:
                continue
            batch_size = len(h)//2
            # sample negative data
            z_pos = torch.vstack([h[batch_size:], h[:batch_size]]) if self.opt.asymmetric_W_pred else h[:batch_size] # B, C_out(* H * W)
            z_neg, rand_index = self.sample_negatives(z_pos, rand_index, cur_device) # n, B, C_out(* H * W)

            # project contextual representation
            c = h_list[self.fb_idx[i]].detach() if self.opt.asymmetric_W_pred else h_list[self.fb_idx[i]][batch_size:]
            c = c.mean(dim=[2, 3])

            # compute score and loss, default averages the spatial neurons
            u_pos = (W(z_pos).mean(dim=[2, 3]) * c) # B, C_out
            n_neg = z_neg.shape[0]
            u_neg = (W(torch.flatten(z_neg, end_dim=1)).mean(dim=[2, 3]).reshape(n_neg, -1, c.shape[1]) * c) # n*B, C_out
            
            loss, rand_fixation = self.compute_loss(u_pos.sum(dim=1), u_neg.sum(dim=2), rand_fixation, cur_device)
            if self.opt.pred_decay > 0:
                loss = loss + self.opt.pred_decay * self.compute_norm(i)
            for key, value in loss_dict.items():
                value[:, i] = loss[key]

        return loss_dict, rand_index, rand_fixation, accuracies
    
    
    def compute_loss(self, u_pos, u_neg, rand_fixation, cur_device):
        if self.contrast_mode=='hinge':
            loss_pos = F.relu(1 - u_pos)
            loss_neg = F.relu(1 + u_neg).mean(dim=0)

            if self.either_pos_or_neg_update:
                if (rand_fixation is None) or (not self.opt.unified_random_sampling):
                    rand_fixation = (torch.rand(len(u_pos), device=cur_device) > 0.5)
                loss = torch.where(rand_fixation, loss_pos, loss_neg)
            else:
                loss = self.opt.pos_coeff * loss_pos + loss_neg
        elif self.contrast_mode=='softhinge':
            loss_pos = -F.logsigmoid(u_pos - 1)
            loss_neg = -F.logsigmoid(-u_neg - 1).mean(dim=0)

            if self.either_pos_or_neg_update:
                if (rand_fixation is None) or (not self.opt.unified_random_sampling):
                    rand_fixation = (torch.rand(len(u_pos), device=cur_device) > 0.5)
                loss = torch.where(rand_fixation, loss_pos, loss_neg)
            else:
                loss = self.opt.pos_coeff * loss_pos + loss_neg
        elif self.contrast_mode=='phyll':
            loss = self.opt.phyll_theta*torch.nn.functional.softplus(u_neg.mean(dim=0) - u_pos, beta=self.opt.phyll_theta) #
            loss_neg = u_neg.mean(dim=0)
            loss_pos = u_pos 
        elif self.contrast_mode=='linear':
            loss_neg = u_neg.mean(dim=0)
            loss_pos = -u_pos 
            loss = u_neg.mean(dim=0) - u_pos
        else:
            raise NotImplementedError

        output = {'loss': loss.mean(), 'loss_pos': loss_pos.mean(), 'loss_neg': loss_neg.mean()}
        return output, rand_fixation



class InfoNCE(nn.Module): # Contrastive Predictive
    def __init__(self, opt, h_dims, proj_kernel=1, proj_stride=1, save_vars=False, diff_layer_pred=False): # in_channels: z, out_channels: c
        super().__init__()
        self.opt = opt
        self.h_dims = [h_dims[-1]] if opt.ete_training else h_dims
        self.negative_samples = self.opt.negative_samples
        self.avoid_same_neg_sample = self.opt.avoid_same_neg_sample
        self.diff_layer_pred = diff_layer_pred
        self.contrast_mode = self.opt.contrast_mode
        self.detach_c = self.opt.detach_c
        self.either_pos_or_neg_update = self.opt.either_pos_or_neg_update
        self.sim_fun = nn.CosineSimilarity(dim=-1, eps=1e-9)
        self.temp = 0.1
        

        if opt.customize_fb_idx is not None:
            self.fb_idx = [int(i)-1 for i in opt.customize_fb_idx.split('-')]
        else:
            self.fb_idx = range(opt.model_splits)
        
        if opt.customize_loss_pool is not None:
            self.pool_module = nn.ModuleList(nn.AvgPool2d(kernel_size=int(w), stride=int(w), padding = 0) for w in opt.customize_loss_pool.split('-'))
        else:
            self.pool_module = None
        self.init_proj(proj_kernel, proj_stride)
        
    def init_proj(self, proj_kernel, proj_stride):

        if self.opt.use_transpose_pred and proj_kernel > 1:
            raise NotImplementedError('have not encounter the scenario for transpose2dConv')
            
        if self.opt.customize_loss_pool is not None:
            self.h_dims = [dims[1] * int(dims[2]/int(w)) * int(dims[3]/int(w)) for w, dims in zip(self.opt.customize_loss_pool.split('-'), self.h_dims)]
        else:
            self.h_dims = [dims[1] for dims in self.h_dims]
        
        in_channels = self.h_dims if self.opt.ete_training else [self.h_dims[i] for i in self.fb_idx]
        out_channels = self.h_dims
        

        self.W_k = nn.ModuleList(
            nn.Linear(c_in, self.opt.low_rank_dim, bias=False) # in_channels: z, out_channels: c
            for c_in in self.h_dims
        )
        
        if self.opt.use_asym_proj_head:
            self.W_k_mirror = nn.ModuleList(
                nn.Linear(c_in, self.opt.low_rank_dim, bias=False) # in_channels: z, out_channels: c
                for c_in in in_channels
            )
        
        self.n_loss = len(self.W_k)
    
    def get_params(self, idx):
        if self.opt.use_asym_proj_head:
            return list(self.W_k[idx].parameters()) +  list(self.W_k_mirror[idx].parameters())
        else:
            return self.W_k[idx].parameters() 
    
    def compute_norm(self, idx):
        if self.opt.use_asym_proj_head:
            raise 0.5* (torch.square(self.W_k[idx].weight).sum() + torch.square(self.W_k_mirror[idx].weight).sum())
        else:
            return torch.square(self.W_k[idx].weight).sum()
             

    def sample_negatives(self, z, rand_index, cur_device):
        if (rand_index is None) or (not self.opt.unified_random_sampling):
            rand_index = torch.randint(z.shape[0], # upper limit: b
                (z.shape[0] * self.negative_samples,), # shape: b, assumes n=1 neg. samples, 
                dtype=torch.long,
                device=cur_device,
            )

        #z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1], z.shape[2], z.shape[3])) # n, B, C, H, W
        z_neg = z[rand_index].reshape((self.negative_samples, z.shape[0], z.shape[1])) # n, B, C, H, W

        return z_neg, rand_index
    
    def process_reps(self, h_list):
        if self.pool_module is not None:
            new_h = [self.pool_module[i](h).flatten(start_dim=1) for i, h in enumerate(h_list)]
            #print('shape {}'.format([h_.shape for h_ in new_h]))
        else:
            new_h = [torch.mean(h, (2,3)) if h.dim()>2 else h for i, h in enumerate(h_list)]
        return new_h

            
    def forward(self, h_list, rand_index=None, rand_fixation=None, idx_range=None, proj_c = False):
        if idx_range is None:
            idx_range = list(range(self.n_loss))
        h_list = self.process_reps(h_list)
        cur_device = h_list[0].get_device() if self.opt.device.type != "cpu" else self.opt.device
        accuracies = torch.zeros(1, self.n_loss, device=cur_device)
        loss_dict = {'loss': torch.zeros(1, self.n_loss, device=cur_device)}
        if self.opt.log_pos_neg:
            loss_dict['loss_pos'] = torch.zeros(1, self.n_loss, device=cur_device)
            loss_dict['loss_neg'] = torch.zeros(1, self.n_loss, device=cur_device)
        for i, (h, W) in enumerate(zip(h_list, self.W_k)):
            if i not in idx_range:
                continue
            batch_size = len(h)//2
            # sample negative data
            z = torch.vstack([h[batch_size:], h[:batch_size]]) if self.opt.asymmetric_W_pred else h[:batch_size] # B, C_out(* H * W)
            wz = W(z)

            # project contextual representation
            c = h_list[self.fb_idx[i]].detach() if self.opt.asymmetric_W_pred else h_list[self.fb_idx[i]][batch_size:]
            wc = self.W_k_mirror[i](c) if self.opt.use_asym_proj_head else W(c)#B, C_in (* H * W) 

            loss, rand_fixation = self.compute_loss(wz, wc, cur_device)
            if self.opt.pred_decay > 0:
                loss['loss'] = loss['loss'] + self.opt.pred_decay * self.compute_norm(i)
            for key, value in loss_dict.items():
                value[:, i] = loss[key]

        return loss_dict, rand_index, rand_fixation, accuracies
    
    
    def compute_loss(self, wz, wc, cur_device):
        #print('projected input shape {}, mean {}, std {}'.format(wz.shape, wz.mean(), wz.std()))

        if self.contrast_mode=='infonce':
            if self.opt.use_asym_proj_head:
                # Find positive example -> batch_size//2 away from the original example
                batch_size = len(wz)//2
                sim_mat1 = self.sim_fun(wz[:batch_size].unsqueeze(1), wc[:batch_size].unsqueeze(0))/self.temp
                sim_mean = sim_mat1.mean()
                sim_pos = torch.diagonal(sim_mat1).mean()
                logprob1 = -F.log_softmax(sim_mat1, dim=1)
                loss1 = torch.diagonal(logprob1).mean()

                sim_mat2 = self.sim_fun(wz[batch_size:].unsqueeze(1), wc[batch_size:].unsqueeze(0))/self.temp
                sim_mean = (sim_mean + sim_mat2.mean())/2
                sim_pos = (sim_pos + torch.diagonal(sim_mat2).mean())/2
                logprob2 = -F.log_softmax(sim_mat2, dim=1)
                loss2 = torch.diagonal(logprob2).mean()

                loss = loss1 + loss2
            else:
                sim_mat = self.sim_fun(wz.unsqueeze(1), wc.unsqueeze(0))/self.temp
                sim_mean = sim_mat.mean()
                sim_pos = torch.diagonal(sim_mat).mean()
                logprob = -F.log_softmax(sim_mat, dim=1)
                loss = torch.diagonal(logprob).mean()
            
        elif self.contrast_mode=='info2nce':
            raise NotImplementedError('original simclr with 2N')
        else:
            raise NotImplementedError

        output = {'loss': loss, 'loss_pos': sim_pos, 'loss_neg': sim_mean}
        return output, None



class DecodeLoss(nn.Module):
    def __init__(self, opt, h_dims, interpolate_mode='bilinear'):
        super(DecodeLoss, self).__init__()

        self.opt = opt
        img_size = self.get_image_size(opt)
        self.layer_channels = [1 if opt.grayscale else 3] + [h_[1] for h_ in h_dims]
        self.init_decoder()

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.loss = nn.MSELoss()

    def get_image_size(self, opt):
        if opt.dataset == 'imagenet':
            return 224
        elif opt.dataset in ['cifar10', 'cifar10_100']:
            return 32
        else:
            return self.opt.random_crop_size

    def init_decoder(self):

        self.decoder = nn.ModuleList([
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)
            for inplanes, outplanes in zip(self.layer_channels[1:], self.layer_channels[:-1])
        ])

        self.n_loss = len(self.decoder)

        return


    def forward(self, h_list, idx_range=None):

        if idx_range is None:
            idx_range = list(range(self.n_loss))
        cur_device = h_list[0].get_device() if self.opt.device.type != "cpu" else self.opt.device
        loss = torch.zeros(1, self.n_loss, device=cur_device)
        for i, (h_in, h_out) in enumerate(zip(h_list[:-1], h_list[1:])):
            in_shape = h_in.shape[2]
            out_shape = h_out.shape[2]
            if in_shape != out_shape:
                h_out = F.interpolate(h_out, size=[in_shape, h_in.shape[3]], mode=self.interpolate_mode, align_corners=True)

            loss_i = self.loss(self.decoder[i](h_out), h_in)
            loss[:, i] = loss_i / h_in.std().item() # scale by the std of the input to normalize the loss

        return loss