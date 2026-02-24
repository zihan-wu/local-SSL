import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import random
from linear_nn import ClappMLP, ClappCNN, compare_fb_grad
from models import LinearDecoder
import yaml
from utils import MultiplyBatchSampler, merge_in_dict
import argparse
import os

DATA_ROOT = '~/Desktop/EPFL/LCN/MNIST/dataset'
SEED = 42
MODEL_SEED = 42
CROP_SIZE = 16
RRC_TASKS = ['clapp_cnn']
VAL_EXPOSE = False
ADD_NOISE = False
COMPARE_GRAD = True
COMPARE_FREQ = 1  
INQUIRED_GRAD_LAYER = 'all'

def seed_everything(seed):  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def build_dataset(contrastive_aug):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_transform = transforms.Compose([   
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.15, 1.0)),
        #transforms.RandomPerspective(0.3),
        transforms.ToTensor(),
        #transforms.GaussianBlur(7),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(scale=(0.04, 0.16)),
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    if contrastive_aug:
        train_ds = torchvision.datasets.MNIST(root=DATA_ROOT, download=True, transform=train_transform)
        test_ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=test_transform)
        decode_train_ds = torchvision.datasets.MNIST(root=DATA_ROOT, download=True, transform=test_transform)
        return train_ds, test_ds, decode_train_ds
    else:
        train_ds = torchvision.datasets.MNIST(root=DATA_ROOT, download=True, transform=mnist_transform)
        test_ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=mnist_transform)
        return train_ds, test_ds
    
def init_model(model_params):
    task = model_params['task']
    if task == 'clapp':
        model = ClappMLP(model_params)
    elif task == 'clapp_cnn':
        model = ClappCNN(model_params)
    else:
        raise NotImplementedError
    return model.to(model_params['device'])

def load_model_weight(model, old_model, chkpt_map):
    # chkpt_map must contain k,v pairs of parameter names.
    # k is the number in the old model, v is (name in the new model, bool of whether to freeze)
    

    for k, v in chkpt_map.items():
        model.get_parameter(v[0]).data = old_model.get_parameter(k).data
        if v[1]:
             model.get_parameter(v[0]).requires_grad = False
    
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    return model, optim_params


def collect_ep_dicts(result_dicts):
    result_stats = {}
    for dict_ep in result_dicts:
        for key in dict_ep:
            if key.endswith('_std'):
                key_orig = key[:-4]
                if key not in result_stats:
                    result_stats[key] = [np.nanstd(dict_ep[key_orig]).item()]
                else:
                    result_stats[key].append(np.nanstd(dict_ep[key_orig]).item())
            else:
                if key not in result_stats:
                    result_stats[key] = [np.nanmean(dict_ep[key]).item()]
                else:
                    result_stats[key].append(np.nanmean(dict_ep[key]).item())
    return result_stats

def exclude_from_wd_and_adaptation(name):
    if 'bn' or 'bias' in name:
        return True
    else:
        return False

def configure_opt(model):
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.99,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups

def train_encoder(model_params, train_args, device):
    
    if model_params['task'] in RRC_TASKS and not model_params['patch_input']:
        train_ds, test_ds, decoder_train_ds = build_dataset(contrastive_aug=True)
        train_ds = torch.utils.data.Subset(train_ds, range(1024))
        test_ds = torch.utils.data.Subset(test_ds, range(1024))
        train_sampler = MultiplyBatchSampler(torch.utils.data.sampler.SequentialSampler(train_ds), batch_size=train_args['bs'], drop_last=True) # RandomSampler
        trainloader = torch.utils.data.DataLoader(train_ds, batch_sampler=train_sampler)
        model_params['input_dim'] = CROP_SIZE * CROP_SIZE
    else:
        train_ds, test_ds = build_dataset(contrastive_aug=False)
        train_ds = torch.utils.data.Subset(train_ds, range(1024))
        test_ds = torch.utils.data.Subset(test_ds, range(1024))
        trainloader = torch.utils.data.DataLoader(train_ds, batch_size=train_args['bs'], shuffle=False)
    testloader = torch.utils.data.DataLoader(train_ds, batch_size=train_args['bs'], shuffle=False)
    
    seed_everything(MODEL_SEED)
    model_params['device'] = device
    model = init_model(model_params)

        
    
    print(model)
    model.freeze_layer(list(range(len(model_params['layer_dim']))))
    print('Model number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Name trainable parameters: {}'.format([name for name, param in model.named_parameters() if param.requires_grad]))
    param_groups = configure_opt(model)
    train_args['opt'] =  torch.optim.Adam(model.parameters(), lr=train_args['lr'], weight_decay=0) #torch.optim.SGD(param_groups, lr=train_args['lr'], momentum=0.8)  # torch.optim.Adam(model.parameters(), lr=train_args['lr'], weight_decay=0) #LARS(torch.optim.SGD(param_groups, lr=train_args['lr'], momentum=0.9))
    optimizer = train_args['opt']
    

    error_sim = []
    weight_grad_sim = []
    bias_grad_sim = []
    result_dicts = [{} for _ in range(train_args['epoch']+1)]
    proj_align_dicts = [{} for _ in range(train_args['epoch']+1)]


    ete_model_params = model_params.copy()
    ete_model_params['layerwise'] = False
    ete_model_params['fb_idx'] = [0] #range(len(model_params['fb_idx']))
    ete_model = init_model(ete_model_params)
    print(ete_model)
    train_args['opt_ete'] =  torch.optim.Adam(ete_model.parameters(), lr=train_args['lr'], weight_decay=0)
    ete_optimizer = train_args['opt_ete']
    ete_model.freeze_layer(list(range(len(model_params['layer_dim']))))
    print('Model number of trainable parameters: {}'.format(sum(p.numel() for p in ete_model.parameters() if p.requires_grad)))
    print('Name trainable parameters: {}'.format([name for name, param in ete_model.named_parameters() if param.requires_grad]))
    with open(model_params['partial_load_dict'], "r") as stream:
        map_dict = yaml.safe_load(stream)


    seed_everything(SEED)
    for x, y in trainloader:
        print('Training on one batch of size {}'.format(x.shape[0]))

        
        # train model last W_pred
        ete_model.train()
        for ep in range(train_args['epoch']):
            ete_optimizer.zero_grad()
            loss_list, optim_loss_list = ete_model.forward_analytical_b(x.to(device))
            # loss_list = model(x.to(device))
            # loss = loss_list.sum()
            # loss.backward()
            # ete_optimizer.step()
            train_loss = loss_list.detach()
            print('ETE Model After Epoch {}, Train loss {}'.format(ep, train_loss))
        model, optimized_params = load_model_weight(model, ete_model, map_dict)
        model.to(device)


        gradloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=train_args['bs'], shuffle=False)
        if COMPARE_GRAD:
            proj_align = model.compute_proj_align()
            proj_align_dicts[0] = merge_in_dict(proj_align_dicts[0], proj_align)
            print('Projection matrix alignment {}'.format(proj_align))
            result_ep = compare_fb_grad(model, gradloader, device, inquired_layer='all')
            result_dicts[0] = merge_in_dict(result_dicts[0], result_ep)


        for ep in range(train_args['epoch']):
            model.train()
            # use the same input

            optimizer.zero_grad()
            loss_list, optim_loss_list = model.forward_analytical_b(x.to(device))
            # loss_list = model(x.to(device))
            # loss = loss_list.sum()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            train_loss = optim_loss_list.detach()

            if (ep + 1) % COMPARE_FREQ == 0:
                print('After Epoch {}, Train loss {}'.format(ep, train_loss))

            if COMPARE_GRAD and (ep + 1) % COMPARE_FREQ == 0:
                proj_align = model.compute_proj_align()
                proj_align_dicts[ep + 1] = merge_in_dict(proj_align_dicts[ep + 1], proj_align)
                print('Projection matrix alignment {}'.format(proj_align))
                result_ep = compare_fb_grad(model, gradloader, device, inquired_layer='all')
                result_dicts[ep + 1] = merge_in_dict(result_dicts[ep + 1], result_ep)

    if COMPARE_GRAD:
        result_stats = collect_ep_dicts(result_dicts)
        proj_align_stats = collect_ep_dicts(proj_align_dicts)
        print('Projection Alignment Stats over epochs:')
        print(proj_align_stats)
        print('Gradient Comparison Stats over epochs:')
        print(result_stats)

        os.makedirs("./stats", exist_ok=True)
        np.save('./stats/proj_align_{}.npy'.format(train_args['model_name']), proj_align_stats)
        np.save('./stats/grad_align_{}.npy'.format(train_args['model_name']), result_stats)

    encoder = model
    encoder.eval()
    
    return encoder,testloader


def exp_rastb(model_params, train_args, save_path):

    print('Current Seed is {}'.format(SEED))
    print(model_params)
    device = torch.device('cpu') # Not tested on GPU, simulation is small enough for just using CPU
    encoder, testloader = train_encoder(model_params, train_args, device)

    
    if save_path is not None and ((model_params['load_path'] is None) or VAL_EXPOSE):
        torch.save(encoder, save_path)
        print('Encoder Saved to {}'.format(save_path))
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run training/evaluation for CLAPP experiments")
    parser.add_argument("--model_config", type=str, default="./configuration_linear.yaml",
                        help="Path to model configuration YAML")
    parser.add_argument("--model_name", type=str, default="128x6_linear_lw_fastb_orthogonal",
                        help="Model name used to build default save path under ./theory_models/")
    parser.add_argument("--train_epochs", "-e", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_bs", type=int, default=32, help="Training batch size")
    parser.add_argument("--train_lr", type=float, default=0.01, help="Training learning rate")
    parser.add_argument("--config_override", type=str, nargs='*', default=[], metavar="KEY=VALUE",
                help="Override model config parameters (e.g., --config_override custom_init=default)")
    return parser.parse_args()

def build_args_from_namespace(ns):
    train_args = {"epoch": ns.train_epochs, "bs": ns.train_bs, "lr": ns.train_lr, "model_name": ns.model_name}
    os.makedirs("theory_models", exist_ok=True)
    save_path = os.path.join("theory_models", f"{ns.model_name}.pth")
    
    # Load model config from YAML
    with open(ns.model_config, "r") as stream:
        model_params = yaml.safe_load(stream)
    
    # Apply model overrides
    for override in ns.config_override:
        key, value = override.split('=')
        model_params[key] = yaml.safe_load(value)
    
    return train_args, save_path, model_params

def parse_and_build():
    ns = parse_arguments()
    return build_args_from_namespace(ns)

if __name__ == "__main__":
    train_args, save_path, model_params = parse_and_build()
    print(train_args)
    exp_rastb(model_params, train_args, save_path)

