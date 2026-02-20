import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UniformProjection(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.uniform = torch.ones((out_dim, in_dim))

    def forward(self, x):
        out = self.scale * x @ self.uniform.T
        return out
    
    @property
    def weight(self):
        return self.scale * self.uniform

class ClappFlexLoss(nn.Module):
    def __init__(self, hparams, layer_dimensions = None) -> None:
        super().__init__()

        self.block_grad = hparams['layerwise']
        self.device = hparams['device']
        self.transition_prob = hparams['fixation_prob']
        self.layer_dim = hparams['layer_dim'] if self.block_grad else [hparams['layer_dim'][-1]]
        self.logsig = nn.LogSigmoid()
        self.margin=1

        if layer_dimensions is None:
            layer_dimensions = self.layer_dim
        
        self.asym_fb = hparams['asym_fb']
        self.sync_labels = hparams['sync_labels']
        self.contrast_mode = hparams['contrast_mode']
        self.b_decay = hparams['b_decay']
        self.fb_idx = hparams['fb_idx']
        if hparams['low_rank'] is not None:
            lr_dim = hparams['low_rank']
            self.Wproj = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_features=d, out_features=lr_dim, bias=False).to(self.device),
                                            nn.Linear(in_features=lr_dim, out_features=layer_dimensions[self.fb_idx[i]], bias=False).to(self.device)
                                            ) for i, d in enumerate(layer_dimensions)])
            self.Wproj[-1] = nn.Linear(in_features=layer_dimensions[-1], out_features=layer_dimensions[self.fb_idx[-1]], bias=False).to(self.device)
        elif getattr(hparams, 'uniform_proj', False):
            # use uniform projection matrices, only scale is trainable
            # each uniform matrix is a, with a being trainable scalar
            self.Wproj = nn.ModuleList([UniformProjection(d, layer_dimensions[self.fb_idx[i]]).to(self.device) for i, d in enumerate(layer_dimensions)])
        else:
            self.Wproj = nn.ModuleList([nn.Linear(in_features=d, out_features=layer_dimensions[self.fb_idx[i]], bias=False).to(self.device) for i, d in enumerate(layer_dimensions)])
        if hparams['asym_fb']:
            if hparams['low_rank'] is not None:
                lr_dim = hparams['low_rank']
                self.Wproj_asym = nn.ModuleList([nn.Sequential(
                                            nn.Linear(in_features=d, out_features=lr_dim, bias=False).to(self.device),
                                            nn.Linear(in_features=lr_dim, out_features=layer_dimensions[self.fb_idx[i]], bias=False).to(self.device)
                                            ) for i, d in enumerate(layer_dimensions)])
                self.Wproj_asym[-1] = nn.Linear(in_features=layer_dimensions[-1], out_features=layer_dimensions[self.fb_idx[-1]], bias=False).to(self.device)
            elif getattr(hparams, 'uniform_proj', False):
                self.Wproj_asym = nn.ModuleList([UniformProjection(d, layer_dimensions[self.fb_idx[i]]).to(self.device) for i, d in enumerate(layer_dimensions)])
            else:
                self.Wproj_asym = nn.ModuleList([nn.Linear(in_features=d, out_features=layer_dimensions[self.fb_idx[i]], bias=False).to(self.device) for i, d in enumerate(layer_dimensions)])
            self.detach_c = True
        else:
            self.detach_c = False

    def get_eff_weights(self, layer):
        # get the effective weights of the projection layers 
        layer = self.Wproj[layer]
        if isinstance(layer, nn.Sequential):
            eff_weights = layer[1].weight @ layer[0].weight
        else:
            eff_weights = layer.weight
        return eff_weights
    
    
    def compute_l2(self, proj):
        # compute the l2 norm of the projection weights
        if isinstance(proj, nn.Sequential):
            l2_norm = torch.square(proj[1].weight @ proj[0].weight).sum()  #torch.square(proj[0].weight).sum() + torch.square(proj[1].weight).sum()
        else:
            l2_norm = torch.square(proj.weight).sum()
        return l2_norm
    
    def get_neg_z(self, z):
        batch_size = len(z)
        current_id = torch.arange(batch_size, device=self.device)
        next_id = (current_id + 1) % batch_size
        #next_id = (current_id + torch.randint(1, batch_size, (batch_size))) % batch_size
        z_new = z.index_select(dim=0, index=next_id)
        return z_new

    def forward(self, z_list, fixed_label=None, return_label=False, return_sim=False, check_pos_only=False):
        # z_list must be a list of hidden representations
        n=len(z_list)
       
        loss = torch.zeros(n, device=self.device, requires_grad=True).clone()
        sim_list = []
        for i, (z, proj, proj_asym) in enumerate(zip(z_list, self.Wproj, self.Wproj_asym)):
            z_top = z_list[self.fb_idx[i]]
            batch_half = int(len(z)/2)
            #z = self.norm[i](z)
            z_ref = [z_top[batch_half:].detach(), z_top[:batch_half].detach()]
            z_ref = torch.vstack(z_ref)
            #z_current, labels, id = self.process_z(z, fixed_label=fixed_label)
            #z_proj = proj(z)
            z_proj = torch.vstack([proj(z[:batch_half]), proj_asym(z[batch_half:])])
            z_neg = self.get_neg_z(z_proj)

            if self.sync_labels:
                fixed_label = fixed_label

            # z_current = torch.clamp(z_current, max=100, min=-100)
            # z_proj = torch.clamp(z_proj, max=100, min=-100)
            #print('context stats {}, z stats{}'.format([z_proj.mean(), z_proj.std()], [z_current.mean(), z_current.std()]))
            # if self.normalize:
            #     #raise NotImplementedError('Not sure how to normalize spatial features')
            #     z_current = F.normalize(z_current, p=2, dim=1)/0.1
            #     z_proj = F.normalize(z_proj, p=2, dim=1)/0.1

            u_pos = (z_proj * z_ref).sum(dim=1)
            u_neg = (z_neg * z_ref).sum(dim=1)
            if return_sim:
                sim_list.append(F.cosine_similarity(z, z_proj, dim=1))


            if self.contrast_mode == 'hinge':
                hinge_loss_pos = (self.margin - u_pos).clamp(min=0)
                hinge_loss_neg = (self.margin + u_neg).clamp(min=0)
                final_loss = hinge_loss_pos + hinge_loss_neg
            elif self.contrast_mode == 'phyll':
                final_loss = torch.nn.functional.softplus(u_neg - u_pos, beta=1.0)
            elif self.contrast_mode == 'logsig':
                loss_pos = -self.logsig(u_pos - self.margin)
                loss_neg = -self.logsig(- u_neg - self.margin)
                final_loss = loss_pos + loss_neg
            elif self.contrast_mode == 'linear':
                final_loss = -u_pos if check_pos_only else u_neg - u_pos
            loss_i = final_loss.mean()
            if self.b_decay > 0:
                loss_i = loss_i + self.b_decay*(self.compute_l2(proj) + self.compute_l2(proj_asym))/2
            #hinge_loss = -self.logsig(score - self.margin)

            loss[i] += loss_i
            #print('scores {} {}, loss {}'.format(score.mean(), score.std(), loss[i]))
        if return_label:
            return loss, fixed_label
        elif return_sim:
            return loss, sim_list
        else:
            return loss  

        
    def forward_analytical_b(self, z_list, fixed_label=None, check_pos_only=False):
        # z_list must be a list of hidden representations
        n=len(z_list)
       
        loss = torch.zeros(n, device=self.device, requires_grad=True).clone()
        optim_loss = torch.zeros(n, device=self.device, requires_grad=True).clone()
        sim_list = []
        for i, (z, proj, proj_asym) in enumerate(zip(z_list, self.Wproj, self.Wproj_asym)):
            
            z_top = z_list[self.fb_idx[i]]
            batch_half = int(len(z)/2)
            #z = self.norm[i](z)
            z_ref = [z_top[batch_half:].detach(), z_top[:batch_half].detach()]
            z_ref = torch.vstack(z_ref)
            #z_current, labels, id = self.process_z(z, fixed_label=fixed_label)
            #z_proj = proj(z)
            z_proj = torch.vstack([proj(z[:batch_half]), proj_asym(z[batch_half:])])
            z_neg = self.get_neg_z(z_proj)

            if self.sync_labels:
                fixed_label = fixed_label

            # z_current = torch.clamp(z_current, max=100, min=-100)
            # z_proj = torch.clamp(z_proj, max=100, min=-100)
            #print('context stats {}, z stats{}'.format([z_proj.mean(), z_proj.std()], [z_current.mean(), z_current.std()]))
            # if self.normalize:
            #     #raise NotImplementedError('Not sure how to normalize spatial features')
            #     z_current = F.normalize(z_current, p=2, dim=1)/0.1
            #     z_proj = F.normalize(z_proj, p=2, dim=1)/0.1

            u_pos = (z_proj * z_ref).sum(dim=1)
            u_neg = (z_neg * z_ref).sum(dim=1)
            
            if self.contrast_mode == 'hinge':
                hinge_loss_pos = (self.margin - u_pos).clamp(min=0)
                hinge_loss_neg = (self.margin + u_neg).clamp(min=0)
                final_loss = hinge_loss_pos + hinge_loss_neg
            elif self.contrast_mode == 'phyll':
                final_loss = torch.nn.functional.softplus(u_neg - u_pos, beta=1.0)
            elif self.contrast_mode == 'logsig':
                loss_pos = -self.logsig(u_pos - self.margin)
                loss_neg = -self.logsig(- u_neg - self.margin)
                final_loss = loss_pos + loss_neg
            elif self.contrast_mode == 'linear':
                final_loss = -u_pos if check_pos_only else u_neg - u_pos
            
        
                        
            #assert self.b_decay > 0, 'analytical b decay requires b_decay > 0'
            #final_loss = final_loss + self.b_decay*(self.compute_l2(proj) + self.compute_l2(proj_asym))/2
            loss_i = final_loss.mean()
            loss[i] += loss_i + self.b_decay*(self.compute_l2(proj).detach() + self.compute_l2(proj_asym).detach())/2
            if (isinstance(proj, nn.Sequential) and not proj[0].weight.requires_grad) or (not isinstance(proj, nn.Sequential) and not proj.weight.requires_grad):
                optim_loss[i] = loss[i]
                continue


            local_optim = torch.optim.LBFGS(list(proj.parameters()) + list(proj_asym.parameters()), lr=0.2, max_iter=1000, line_search_fn='strong_wolfe')
            def compute_loss():
                local_optim.zero_grad()
                z_proj = torch.vstack([proj(z[:batch_half].detach()), proj_asym(z[batch_half:].detach())])
                z_neg = self.get_neg_z(z_proj)
                u_neg = (z_neg * z_ref.detach()).sum(dim=1)
                u_pos = (z_proj * z_ref.detach()).sum(dim=1)
                if self.contrast_mode == 'hinge':
                    hinge_loss_pos = (self.margin - u_pos).clamp(min=0)
                    hinge_loss_neg = (self.margin + u_neg).clamp(min=0)
                    final_loss = hinge_loss_pos + hinge_loss_neg
                elif self.contrast_mode == 'phyll':
                    final_loss = torch.nn.functional.softplus(u_neg - u_pos, beta=1.0)
                elif self.contrast_mode == 'logsig':
                    loss_pos = -self.logsig(u_pos - self.margin)
                    loss_neg = -self.logsig(- u_neg - self.margin)
                    final_loss = loss_pos + loss_neg
                elif self.contrast_mode == 'linear':
                    final_loss = u_neg - u_pos

                total_loss = final_loss.mean() + self.b_decay*(self.compute_l2(proj) + self.compute_l2(proj_asym))/2
                total_loss.backward(retain_graph=True)
                return total_loss
            
            local_optim.step(compute_loss)
            # proj.weight.data = torch.autograd.grad(-loss_i, proj.weight, retain_graph=True)[0].detach()/(self.b_decay)
            # proj_asym.weight.data = torch.autograd.grad(-loss_i, proj_asym.weight, retain_graph=True)[0].detach()/(self.b_decay)
            
            z_proj_optim = torch.vstack([proj(z[:batch_half]), proj_asym(z[batch_half:])])
            z_neg_optim = self.get_neg_z(z_proj_optim)
            u_neg_optim = (z_neg_optim * z_ref).sum(dim=1)
            u_pos_optim = (z_proj_optim * z_ref).sum(dim=1)
            
            # Compute new loss
            if self.contrast_mode == 'hinge':
                hinge_loss_pos = (self.margin - u_pos_optim).clamp(min=0)
                hinge_loss_neg = (self.margin + u_neg_optim).clamp(min=0)
                new_loss = hinge_loss_pos + hinge_loss_neg
            elif self.contrast_mode == 'phyll':
                new_loss = torch.nn.functional.softplus(u_neg_optim - u_pos_optim, beta=1.0)
            elif self.contrast_mode == 'logsig':
                loss_pos = -self.logsig(u_pos_optim - self.margin)
                loss_neg = -self.logsig(- u_neg_optim - self.margin)
                new_loss = loss_pos + loss_neg
            elif self.contrast_mode == 'linear':
                new_loss = -u_pos_optim if check_pos_only else u_neg_optim - u_pos_optim

            optim_loss[i] = new_loss.mean() + self.b_decay*(self.compute_l2(proj) + self.compute_l2(proj_asym))/2

            #print('scores {} {}, loss {}'.format(score.mean(), score.std(), loss[i]))
        return loss, optim_loss

    def forward_best_grad_pred(self, z_list, fixed_label=None, return_label=False, return_sim=False, check_pos_only=False):

        n=len(z_list)
       
        loss = torch.zeros(n, device=self.device, requires_grad=True).clone()
        sim_list = []
        for i in range(n-1, -1, -1):
            (z, proj, proj_asym) = (z_list[i], self.Wproj[i], self.Wproj_asym[i])
            if i < n-1:
                ideal_grad_pred = torch.autograd.grad(loss[-1], z, retain_graph=True)[0]
                z = z.detach()
                z.requires_grad = True
            z_top = z_list[self.fb_idx[i]]
            batch_half = int(len(z)/2)
            #z = self.norm[i](z)
            z_ref = [z_top[batch_half:].detach(), z_top[:batch_half].detach()]
            z_ref = torch.vstack(z_ref)
            #z_current, labels, id = self.process_z(z, fixed_label=fixed_label)
            #z_proj = proj(z)
            z_proj = torch.vstack([proj(z[:batch_half]), proj_asym(z[batch_half:])])
            z_neg = self.get_neg_z(z_proj)

            if self.sync_labels:
                fixed_label = fixed_label

            # z_current = torch.clamp(z_current, max=100, min=-100)
            # z_proj = torch.clamp(z_proj, max=100, min=-100)
            #print('context stats {}, z stats{}'.format([z_proj.mean(), z_proj.std()], [z_current.mean(), z_current.std()]))
            # if self.normalize:
            #     #raise NotImplementedError('Not sure how to normalize spatial features')
            #     z_current = F.normalize(z_current, p=2, dim=1)/0.1
            #     z_proj = F.normalize(z_proj, p=2, dim=1)/0.1

            u_pos = (z_proj * z_ref).sum(dim=1)
            u_neg = (z_neg * z_ref).sum(dim=1)
            if return_sim:
                sim_list.append(F.cosine_similarity(z, z_proj, dim=1))


            if self.contrast_mode == 'hinge':
                hinge_loss_pos = (self.margin - u_pos).clamp(min=0)
                hinge_loss_neg = (self.margin + u_neg).clamp(min=0)
                final_loss = hinge_loss_pos + hinge_loss_neg
            elif self.contrast_mode == 'phyll':
                final_loss = torch.nn.functional.softplus(u_neg - u_pos, beta=1.0)
            elif self.contrast_mode == 'logsig':
                loss_pos = -self.logsig(u_pos - self.margin)
                loss_neg = -self.logsig(- u_neg - self.margin)
                final_loss = loss_pos + loss_neg
            elif self.contrast_mode == 'linear':
                final_loss = -u_pos if check_pos_only else u_neg - u_pos
            if self.b_decay > 0:
                final_loss = final_loss + self.b_decay*(self.compute_l2(proj) + self.compute_l2(proj_asym))/2
            #hinge_loss = -self.logsig(score - self.margin)
            if i == n-1:
                loss[i] += final_loss.mean()
                proj.weight.grad = torch.autograd.grad(loss[i], proj.weight, retain_graph=True)[0]
                proj_asym.weight.grad = torch.autograd.grad(loss[i], proj_asym.weight, retain_graph=True)[0]
            else:
                loss_i = final_loss.mean()
                grad_pred = torch.autograd.grad(loss_i, z, retain_graph=True, create_graph=True)[0]

                loss[i] += ((ideal_grad_pred - grad_pred)**2).sum() # #(1 - F.cosine_similarity(ideal_grad_pred, grad_pred, dim=-1)).sum() #((ideal_grad_pred - grad_pred)**2).sum()
                proj.weight.grad = torch.autograd.grad(loss[i], proj.weight, retain_graph=True)[0]
                proj_asym.weight.grad = torch.autograd.grad(loss[i], proj_asym.weight, retain_graph=True)[0]
            #print('scores {} {}, loss {}'.format(score.mean(), score.std(), loss[i]))
        if return_label:
            return loss, fixed_label
        elif return_sim:
            return loss, sim_list
        else:
            return loss  

