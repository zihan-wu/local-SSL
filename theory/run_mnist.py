import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import random
from models import compare_fb_grad, CLAPP_FB, LinearDecoder
import yaml
from utils import merge_in_dict

DATA_ROOT = '/Users/zihanwu/Desktop/EPFL/LCN/MNIST/dataset'
SEED = 42
MODEL_SEED = 42
VAL_EXPOSE = False
ADD_NOISE = False
COMPARE_GRAD = True
SCALE_PRED = False
INQUIRED_GRAD_LAYER = 'all' #'layer_0' #
MASK_ANALYSIS = None #'per_batch'
CROP_SIZE = 16

def seed_everything(seed):  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def build_dataset(contrastive_aug):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_transform = transforms.Compose([   
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.25, 1.0)),
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

    if ADD_NOISE:
        mnist_transform = transforms.Compose([mnist_transform, AddGaussianNoise(0, 1)])
        print('NOISE Added to Decoding Performance')

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
    if task == 'clapp_fb':
        model = CLAPP_FB(model_params)
    else:
        raise NotImplementedError
    return model.to(model_params['device'])

def load_model_weight(model, chkpt_path, chkpt_map):
    # chkpt_map must contain k,v pairs of parameter names.
    # k is the number in the old model, v is (name in the new model, bool of whether to freeze)
    
    old_model = torch.load(chkpt_path)
    print('Old Model loaded from {} '.format(chkpt_path))
    for k, v in chkpt_map.items():
        model.get_parameter(v[0]).data = old_model.get_parameter(k).data
        if v[1]:
             model.get_parameter(v[0]).requires_grad = False
    
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Paramaters being optimized : {}'.format(optim_params))
    return model, optim_params

def unfreeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Unfreeze whole model. Parameters being optimized : {}'.format(optim_params))
    return model

def decode_validation(encoder, decoder, testloader, device, i_layer=-1):
    decoder.eval()
    total_loss = 0
    all_logit = []
    all_target = []
    for x, y in tqdm(testloader):
        all_target.append(y.to(device))
        with torch.no_grad():
            h = encoder.encode(x.to(device))[i_layer]
            loss, y_pred = decoder(h, y.to(device))
        total_loss += loss
        all_logit.append(y_pred)
    
    all_logit = torch.vstack(all_logit)
    all_pred = torch.argmax(all_logit, 1)
    all_target = torch.hstack(all_target)
    total_loss = total_loss/len(testloader)
    test_acc = sum(all_pred == all_target)/len(all_pred)
    return total_loss.item(), test_acc.item()


def train_encoder(model_params, train_args, eval_args, device):
    train_ds, test_ds = build_dataset(contrastive_aug=False)
    decode_trainloader = torch.utils.data.DataLoader(train_ds, batch_size=eval_args['bs'], shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=train_args['bs'], shuffle=True)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=train_args['bs'], shuffle=False)
    
    if VAL_EXPOSE:
        trainloader = testloader
        train_args['epoch'] = 1




    seed_everything(MODEL_SEED)
    model_params['device'] = device
    model = init_model(model_params)

    if model_params['partial_load_path'] is not None:
        with open(model_params['partial_load_dict'], "r") as stream:
            map_dict = yaml.safe_load(stream)
        model, optimized_params = load_model_weight(model, model_params['partial_load_path'], map_dict)
        if model_params['task'] == 'clapp_fa':
            print('Re-initialize feedback weights')
            model.init_fb_proj(ideal=True, trainable=False)
        model.to(device)
    
    if model_params['load_path'] is not None:
        encoder = torch.load(model_params['load_path'])
        print('Model loaded from {} '.format(model_params['load_path']))
        if not VAL_EXPOSE:
            encoder.eval()
            print(encoder)
            return encoder, decode_trainloader,testloader
        else:
            model = encoder
    
    print(model)
    print('Model number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Name trainable parameters: {}'.format([name for name, param in model.named_parameters() if param.requires_grad]))
    train_args['opt'] =  torch.optim.Adam(model.parameters(), lr=train_args['lr'], weight_decay=0)
    optimizer = train_args['opt']
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eval_args['epoch'], eta_min=1e-5)# 
    #scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)# torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1) 

    seed_everything(SEED)
    
    error_sim = []
    weight_grad_sim = []
    bias_grad_sim = []
    result_dict = {}
    proj_align_dict = {}
    train_loss_history = []


    if COMPARE_GRAD:
        proj_align = model.compute_proj_align()
        proj_align_dict = merge_in_dict(proj_align_dict, proj_align)
        print('Projection matrix alignment {}'.format(proj_align))
        if INQUIRED_GRAD_LAYER == 'all':
            result_ep = compare_fb_grad(model, testloader, device, inquired_layer=INQUIRED_GRAD_LAYER, analyze_mask=MASK_ANALYSIS)
            result_dict = merge_in_dict(result_dict, result_ep)
        else:
            err_sim, weight_sim, bias_sim = compare_fb_grad(model, testloader, device, inquired_layer=INQUIRED_GRAD_LAYER, analyze_mask=MASK_ANALYSIS)
            error_sim.append(err_sim)
            weight_grad_sim.append(weight_sim)
            bias_grad_sim.append(bias_sim)
    


    for ep in range(train_args['epoch']):
        print(f'Training Epoch {ep}')
        model.train()
        train_loss = 0
        for x, y in tqdm(trainloader):
            optimizer.zero_grad()
            if model_params['alignment_loss'] is None:
                loss = model(x.to(device))
            else:
                loss, loss_b_out = model(x.to(device))
            if model_params['task'] not in ['clapp_dfa', 'clapp_fa']:
                if len(loss) > 1:
                    loss_sum = loss.sum()
                    loss_sum.backward()
                else:
                    loss.backward()
                if model_params['task'] == 'clapp_fb' and SCALE_PRED:
                    model.scale_pred_grad(0.01) #model_params['a_amp']
                if model_params['train_fb_with_grad']:
                    fb_loss = model.forward_fb_with_grad(x.to(device))
            optimizer.step()
            if model_params['scale_fb_weight']:
                model.scale_fb_weight()
            train_loss += loss.detach()
        train_loss = train_loss/len(trainloader)
        print('After Epoch {}, Train loss {}'.format(ep, train_loss))

        scheduler.step()
        if COMPARE_GRAD:
            proj_align = model.compute_proj_align()
            proj_align_dict = merge_in_dict(proj_align_dict, proj_align)
            print('Projection matrix alignment {}'.format(proj_align))
            if INQUIRED_GRAD_LAYER == 'all':
                result_ep = compare_fb_grad(model, testloader, device, inquired_layer=INQUIRED_GRAD_LAYER, analyze_mask=MASK_ANALYSIS)
                result_dict = merge_in_dict(result_dict, result_ep)
            else:
                err_sim, weight_sim, bias_sim = compare_fb_grad(model, testloader, device, inquired_layer=INQUIRED_GRAD_LAYER, analyze_mask=MASK_ANALYSIS)
                error_sim.append(err_sim)
                weight_grad_sim.append(weight_sim)
                bias_grad_sim.append(bias_sim)
        

        train_loss_history.append(train_loss)

    if COMPARE_GRAD:
        print(proj_align_dict)
        if INQUIRED_GRAD_LAYER == 'all':
            print(result_dict)
        else:
            print(err_sim)
            print(weight_sim)
        np.save('./stats/proj_align_{}.npy'.format(MODEL_NAME), proj_align_dict)
        if MASK_ANALYSIS is not None:
            np.save('./stats/grad_align_{}_with_mask_{}.npy'.format(MODEL_NAME, MASK_ANALYSIS), result_dict)
        else:
            np.save('./stats/grad_align_{}.npy'.format(MODEL_NAME), result_dict)

    np.save('./stats/train_loss_{}.npy'.format(MODEL_NAME), train_loss_history)
    encoder = model
    encoder.eval()

    testloader = torch.utils.data.DataLoader(test_ds, batch_size=eval_args['bs'], shuffle=False)
    
    return encoder, decode_trainloader,testloader


def train_decoder(encoder, eval_args, trainloader, testloader, device):
    decode_layer = eval_args['layer'][0]
    dim_layer = eval_args['layer'][1]
    print('Using decode layer {}, dim {}'.format(decode_layer, dim_layer))
    seed_everything(SEED)
    decoder = LinearDecoder(encoder.layer_dim[dim_layer], 10).to(device)
    eval_args['opt'] = torch.optim.Adam(decoder.parameters(), lr=eval_args['lr'], weight_decay=0) #torch.optim.SGD(decoder.parameters(), lr=eval_args['lr'], momentum=0.9)
    eval_optimizer = eval_args['opt']

    test_loss, test_acc = decode_validation(encoder, decoder, testloader, device, decode_layer)
    print('Initial Accuracy {}'.format(test_acc))
    best_accu = test_acc
    for ep in range(eval_args['epoch']):
        print(f'Training Decoder Epoch {ep}')
        decoder.train()
        for x, y in tqdm(trainloader):
            with torch.no_grad():
                h = encoder.encode(x.to(device))[decode_layer]
            loss, y_pred = decoder(h, y.to(device))
            eval_optimizer.zero_grad()
            loss.backward()
            eval_optimizer.step()
        
        test_loss, test_acc = decode_validation(encoder, decoder, testloader, device, decode_layer)
        if test_acc > best_accu:
            best_accu=test_acc
        print('After Epoch {}, Test loss {}, Test Accuracy {}'.format(ep, test_loss, test_acc))
    print('Best Score is {}'.format(best_accu))
    return best_accu


def main(model_config, train_args, eval_args, save_path):
    with open(model_config, "r") as stream:
        model_params = yaml.safe_load(stream)

    print('Current Seed is {}'.format(SEED))
    print(model_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decode_trainloader, testloader = train_encoder(model_params, train_args, eval_args, device)

    
    if save_path is not None and ((model_params['load_path'] is None) or VAL_EXPOSE):
        torch.save(encoder, save_path)
        print('Encoder Saved to {}'.format(save_path))
    if eval_args is not None:
        best_accu = train_decoder(encoder, eval_args, decode_trainloader, testloader, device)
        return best_accu
    return None

if __name__ == "__main__":
    train_args = {'epoch': 20, 'bs': 32, 'lr': 5e-5} # 1e-5 for simclr, 5e-5 for patch mlp #0.02 for mae (0.04 if normalized), 0.05 for clapp, 4.8 for simclr but 50 ep
    eval_args = {'layer': [-1, -1], 'epoch': 20, 'bs': 32, 'lr': 0.002} # 0.005 for simclr, 0.002 for patch 5e-3 for mae, 3e-3 for clapp, 1e-3 for simclr
    #train_args = {'epoch': 20, 'bs': 32, 'lr':2e-4}
    #eval_args = {'epoch': 20, 'bs': 32, 'lr': 1e-3}
    print(train_args, eval_args)
    MODEL_NAME = 'clapp_ete_512x6_logsigl2_seed42' #'clapp_fb_512x6_train_fb_with_grad_logsigl2_repro_seed42' #'clapp_vgg_crop16_128x4_lw_nopool_logsigl2_seed42'
    main('configuration_dfa.yaml', train_args, eval_args, 'models/{}.pth'.format(MODEL_NAME)) # finetune_supervised_ete/25ep_1024_4x3_weightshare_noresg_leaveg_all_layer_tuneg_seed42.pth'
