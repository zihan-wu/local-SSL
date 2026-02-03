# Code to compare numerically the updates stemming from (i) CLAPP learning rules and (ii) CLAPP loss + autodiff in pytorch

# This is tested for 
# 1) using the same (single) negative everywhere (local sampling): --sample_negs_locally --sample_negs_locally_same_everywhere
# 2) not using W_retro for the moment (i.e. NOT --asymmetric_W_pred)

# E.g. the below tests hold for a randomly initialised network at the first epoch of training:
# mkdir ./logs/CLAPP_init/
# python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_init --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 1 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --start_epoch 0 --model_path ./logs/CLAPP_init/ --save_vars_for_update_calc 3 --batch_size 4

# Tests also held for later points in training, e.g. after ~600 epochs:
# To reproduce this, the respective simulations first need to be run/created (running 'CLAPPVision.vision.main_vision') with respective command line options
# E.g. for CLAPP-s:
# python -m CLAPPVision.vision.main_vision --download_dataset --save_dir CLAPP_1 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere
# python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_1 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --start_epoch 599 --model_path ./logs/CLAPP_1/ --save_vars_for_update_calc 3 --batch_size 4
# or for CLAPP:
# python -m CLAPPVision.vision.main_vision --download_dataset --save_dir CLAPP_2 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --either_pos_or_neg_update
# python -m CLAPPVision.vision.compare_updates --download_dataset --save_dir CLAPP_2 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --either_pos_or_neg_update --start_epoch 599 --model_path ./logs/CLAPP_2/ --save_vars_for_update_calc 3 --batch_size 4

################################################################################

# Switch to model with train_ssl_new:
# 1. change the load_model_and_optimizer 
# 2. add the configuration loading code

from shutil import which
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.random import choice
import time
import os
import code
import matplotlib.pyplot as plt
import copy
import json
import glob
import re
from itertools import islice


## own modules
from vision.data import get_dataloader
from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from vision.utils import logger, utils



def train_iter(opt, model, model_input, label, ete=False):
    if ete:
        model.module.ete_training = True
    else:
        model.module.ete_training = opt.ete_training

    model.zero_grad()
    cur_train_module = opt.train_module
    
    label = label.to(opt.device)
    loss_dict, h, accuracies = model(model_input, label, n=cur_train_module)
    loss = torch.mean(loss_dict['loss'], 0) # take mean over outputs of different GPUs
    print(loss)
    if ete:
        loss[-1].backward() # only backpropagate the last output, as this is the one used for training
    else:
        loss.sum().backward()


def _load_activations(opt, layer, k):
    which_update = torch.load(os.path.join(opt.model_path, 'saved_which_update_layer_'+str(layer)), map_location=torch.device('cpu'))
    
    (context, z_p, z_n, rand_index) = torch.load(os.path.join(opt.model_path, 'saved_c_and_z_layer_'+str(layer)+'_k_'+str(k)), map_location=torch.device('cpu'))
    context = context.squeeze(-2)
    # all size: y (red.), x, b, c

    (loss_p, loss_n) = torch.load(os.path.join(opt.model_path, 'saved_losses_layer_'+str(layer)+'_k_'+str(k)), map_location=torch.device('cpu'))
    # b, 1, y (red.), x
    dloss_p = - torch.sign(loss_p.squeeze(1).permute(1, 2, 0))
    dloss_n = torch.sign(loss_n.squeeze(1).permute(1, 2, 0))
    # y (red.), x, b

    return which_update, context, z_p, z_n, rand_index, dloss_p, dloss_n



def get_Wpred(opt, model, layer):
    

    grad_Wpred = model.module.loss.W_k[layer-1].weight.grad.squeeze().clone().detach().to('cpu')
    # model.module.model[0][layer-1].loss_mirror.W_k[0].weight.grad

    return grad_Wpred


def get_W_ff(opt, model, layer, sub_layer=0):


    grad_W_ff = model.module.encoder[layer-1].model[sub_layer].weight.grad.squeeze().clone().detach().to('cpu')
    # model.module.model[0][layer-1].model[0].bias.grad

    return grad_W_ff

def load_model_weight(model, chkpt_path, model_num, chkpt_map, flexible_reload=True):
    # chkpt_map must contain k,v pairs of parameter names.
    # k is the number in the old model, v is (name in the new model, bool of whether to freeze)
    # Currently only support loading a whole model (if saved by individual encoder, please first compose into a whole model)
    
    for keys, map_dict in chkpt_map.items():
        if flexible_reload:
            files=glob.glob(os.path.join(
                                chkpt_path,
                                "model_{}_*.ckpt".format(keys)))
            available_ckpt = np.sort(np.array([int(re.findall("[-+]?\d+", f)[-1]) for f in files]))
            print('available ckpt to load from model_{}: {}'.format(keys, available_ckpt))
            actual_model_num = np.extract(available_ckpt <= int(model_num), available_ckpt)[-1] 
        else:
            actual_model_num = model_num
        
        sub_model_path = os.path.join(
                            chkpt_path,
                            "model_{}_{}.ckpt".format(keys, actual_model_num),
                        )
        old_model_chkpt = torch.load(sub_model_path)
        print('Old Model loaded from {} '.format(sub_model_path))
        for k, v in map_dict.items():
            model.get_parameter(v[0]).data = old_model_chkpt[k]
            model.get_parameter(v[0]).requires_grad = True
    
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Paramaters being optimized : {}'.format(optim_params))
    print('Number of Parameters being optimized: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model, optim_params
    
    
def get_layer_pairs(splits, n_layer=6):
    sub_layer_inds = [0, 2, 5, 7, 10, 13]
    if splits == 6:
        return [(k, 0) for k in range(1, n_layer+1)]
    elif splits == 1:
        return [(1, k) for k in sub_layer_inds]
    else:
        raise NotImplementedError('only supporting split 1 or 6')

##############################################################################################################

if __name__ == "__main__":
    
    #toyex()
    opt = arg_parser.parse_args()
    opt.train_ds_no_shuffle = True
    layer_type = 'layerwise'
    opt.grayscale = False

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    # with open(os.path.join(opt.model_path, 'config.json'), 'r') as f:
    #     opt.config = json.load(f)

    model, optimizer = load_vision_model.load_ssl_model_and_optimizer(opt, reload_model=True, calc_loss=True)
    print(model)


    if opt.batch_size != opt.batch_size_multiGPU:
        raise Exception("Manual update comparison only supported for 1 GPU. Please use only 1 GPU")
    
    utils.seed_everything(opt.seed)
    train_loader, _, _, _, _, _ = get_dataloader.get_paried_dataloader(opt)

    subloader = islice(train_loader, 10)

    reload_index_folder = opt.reload_index_path
    folder_name = 'gradients_{}'.format(opt.model_num)
    ete_folder_name = 'ete_gradients_{}'.format(opt.model_num)

    gradient_folder = os.path.join(opt.model_path, folder_name)
    ete_gradient_folder = os.path.join(opt.model_path, ete_folder_name)
    if not os.path.exists(gradient_folder):
        os.makedirs(gradient_folder)
        os.makedirs(ete_gradient_folder)
        for l in range(6):
            os.makedirs(os.path.join(gradient_folder, 'layer_{}'.format(l+1)))
            os.makedirs(os.path.join(ete_gradient_folder, 'layer_{}'.format(l+1)))

    layer_inds = get_layer_pairs(splits = opt.model_splits)
    for idx, (batch_img, label) in enumerate(subloader):
        print(label)
        model_input = [img.to(opt.device) for img in batch_img]
        print('loading model again for layerwise calc')
        model, optimizer = load_vision_model.load_ssl_model_and_optimizer(opt, reload_model=True, calc_loss=True)
        model.module.opt.reload_index_path = os.path.join(reload_index_folder, 'batch_{}.pt'.format(idx))
        
        # perform one training iter and save reps etc.
        train_iter(opt, model, model_input, label)
        for k, (l, s_l) in enumerate(layer_inds):
            grad_W_ff = get_W_ff(opt, model, l, s_l)
            print('grad mean {}, std {}'.format(torch.mean(grad_W_ff), torch.std(grad_W_ff)))
            torch.save(grad_W_ff, os.path.join(gradient_folder, 'layer_{}'.format(k+1), 'batch_{}.pt'.format(idx)))

        print('loading model again for ete calc')
        model, optimizer = load_vision_model.load_ssl_model_and_optimizer(opt, reload_model=True, calc_loss=True)
        model.module.opt.reload_index_path = os.path.join(reload_index_folder, 'batch_{}.pt'.format(idx))
        train_iter(opt, model, model_input, label, ete=True)
        for k, (l, s_l) in enumerate(layer_inds):
            grad_W_ff = get_W_ff(opt, model, l, s_l)
            print('ete_grad mean {}, std {}'.format(torch.mean(grad_W_ff), torch.std(grad_W_ff)))
            torch.save(grad_W_ff, os.path.join(ete_gradient_folder, 'layer_{}'.format(k+1), 'batch_{}.pt'.format(idx)))


    # check equivalence between grads and updates for Wpred
    # print("checking equivalence between grads and updates for Wpred...")
    # for k in range(1,6):
    #     grad_Wpred = get_Wpred(opt, model, k, layer=l)
    
    # check equivalence between grads and updates for W_ff
    
    