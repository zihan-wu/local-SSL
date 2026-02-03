import torch
import time
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from vision.data import get_dataloader
from vision.utils import logger, utils, opt_scheduler
import wandb
import os
import json

WANDB = False # set to True if you want to use wandb for logging, otherwise it will just print logs to terminal and save them in log files

def train(opt, model, train_loader, optimizer, logs):

    # scheduler = LinearWarmupCosineAnnealingLR(optimizer,
    #                                         warmup_start_lr=self.hparams.start_lr,
    #                                         eta_min=self.hparams.final_lr,
    #                                         warmup_epochs=self.hparams.warmup_epochs,
    #                                         max_epochs=self.hparams.max_epochs)
    if opt.use_scheduler:
        if opt.dataset == 'imagenet':
            sche = opt_scheduler.CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=opt.num_epochs + opt.start_epoch,
                                                        max_lr=opt.learning_rate, min_lr=1e-5, warmup_steps=5, last_epoch=opt.start_epoch-1)
            optimizer = opt_scheduler.LARS(optimizer)
        else:
            # sche = opt_scheduler.CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=opt.num_epochs + opt.start_epoch, 
            #                                             max_lr=opt.learning_rate, min_lr=1e-6, warmup_steps=5, last_epoch=opt.start_epoch-1)
            sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs + opt.start_epoch, eta_min=1e-6)
    

    total_step = len(train_loader)

    print_idx = 375
    assert opt.module_layer == -1, 'during training, must account all layers in each block'

    starttime = time.time()
    cur_train_module = opt.train_module
    if logs is not None:
        logs.create_log(model, epoch=-1, optimizer=optimizer)

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        
        if opt.distr_strategy == 'ddp':
            train_loader.sampler.set_epoch(epoch)

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_neg_epoch = [0 for i in range(opt.model_splits)]
        loss_pos_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        step_start_time = time.time()
        cum_data_time = step_start_time - step_start_time
        cum_it_time = cum_data_time
        # loop over batches in train_loader
        for step, (batch_img, label) in enumerate(train_loader):

            cum_data_time = cum_data_time + time.time() - step_start_time
            if (step % print_idx == 0) or (step == len(train_loader)-1):
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time from last log(s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

                print(
                    "Epoch [{}/{}], Step [{}/{}], Data Time (s): {:.1f}, Computation Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cum_data_time,
                        cum_it_time,
                    )
                )

                starttime = time.time()

            it_start_time = time.time()
            
            model_input = [img.to(opt.device) for img in batch_img]
            label = label.to(opt.device)
            model.zero_grad()

            # forward pass through whole model (loop over modules within model forward)
            if opt.train_fb_with_grad:
                loss_dict, fb_loss_dict, h, accuracies = model(model_input, label, n=cur_train_module, just_fb_grad=True)
                fb_loss = torch.mean(fb_loss_dict['loss'], 0)
                for i in range(cur_train_module):
                    #print('OuterLoop: Layer {} feedback loss {}, with weight grad norm {}'.format(i, fb_loss[i].item(), model.module.loss.W_k[i].weight.grad.norm().item()))
                    assert model.module.encoder[i].model[0].weight.grad is None, 'encoder weights should not require grad when training fb with grad'
            else:
                loss_dict, h, accuracies = model(model_input, label, n=cur_train_module)

            loss = torch.mean(loss_dict['loss'], 0) # take mean over outputs of different GPUs
            if opt.log_pos_neg:
                loss_pos = torch.mean(loss_dict['loss_pos'], 0) # take mean over outputs of different GPUs
                loss_neg = torch.mean(loss_dict['loss_neg'], 0) # take mean over outputs of different GPUs
            if opt.contrain_norm:
                loss_norm = torch.mean(loss_dict['loss_norm'], 0)
            if opt.decode_loss is not None:
                loss_decode = torch.mean(loss_dict['loss_decode'], 0)
            #loss_aux = torch.mean(loss_aux, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracies, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                loss = loss[:cur_train_module]
                #raise ValueError("Training intermediate modules is not tested!")

            # loop through the losses of the modules and do gradient descent
            wandb_log = {'epoch': epoch, 'step': step}
            
            if not opt.train_fb_with_grad:
                loss.sum().backward()

            #for i in range(cur_train_module):
                #print('Layer {} loss {}, with feedforward grad norm {}'.format(i, loss[i].item(), model.module.encoder[i].model[0].weight.grad.norm().item()))
                #assert model.module.encoder[i].model[0].weight.grad is None, 'encoder weights should not require grad when training fb with grad'
                #print('Layer {} feedback loss {}, with weight grad norm {}'.format(i, fb_loss[i].item(), model.module.loss.W_k[i].weight.grad.norm().item()))

            optimizer.step()
            cum_it_time = cum_it_time + time.time() - it_start_time
            for idx in range(len(loss)):
                # if len(loss) == 1 and opt.model_splits != 1:
                #     idx = cur_train_module - 1

                # add optional optimizer step here

                
                if (opt.distr_strategy != 'ddp' or opt.device_rank == 0):
                # We still output normal (ungated) loss for printing and plotting
                    print_loss = loss[idx].item()

                    if opt.all_layer and not(opt.pool_layer_loss) and idx > 0:
                        print_loss = print_loss/len(model.module.encoder[idx].model)
                    # elif model.module.encoder[idx].use_loss_g:
                    #     print_loss *= 0.5

                    print_acc = accuracy[idx].item()

                    if opt.reg_loss is not None:
                        raise NotImplementedError
                        # print_reg_loss = loss_aux[idx].item()
                        # loss_aux_epoch[idx] += print_reg_loss
                        # print_loss = print_loss - print_reg_loss
                        # wandb_log['reg_loss_{}'.format(idx)] = print_reg_loss 
                    if opt.log_pos_neg:
                        print_neg_loss = loss_neg[idx].item()
                        print_pos_loss = loss_pos[idx].item()
                        loss_neg_epoch[idx] += print_neg_loss
                        loss_pos_epoch[idx] += print_pos_loss
                        wandb_log['pos_loss_{}'.format(idx)] = print_pos_loss 
                        wandb_log['neg_loss_{}'.format(idx)] = print_neg_loss 
                    if opt.train_fb_with_grad:
                        wandb_log['fb_loss_{}'.format(idx)] = fb_loss[idx].item()
                    if opt.contrain_norm:
                        wandb_log['loss_norm_{}'.format(idx)] = loss_norm[idx].item() 
                    if opt.decode_loss is not None:
                        wandb_log['loss_decode_{}'.format(idx)] = loss_decode[idx].item()
                    loss_epoch[idx] += print_loss
                    loss_updates[idx] += 1
                    wandb_log['loss_{}'.format(idx)] = print_loss
                    wandb_log['acc_{}'.format(idx)] = print_acc
                    wandb_log['lr_{}'.format(idx)] = optimizer.param_groups[0]['lr']

                    if step % print_idx == 0:
                        print("\t \t Loss_{}: \t \t {:.4f}".format(idx, print_loss))
                        if opt.loss == 1:
                            print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))
        

            
            if WANDB and (opt.distr_strategy != 'ddp' or opt.device_rank == 0):
                wandb.log(wandb_log)

            
            step_start_time = time.time()
        
        if opt.use_scheduler:
            sche.step()

        if logs is not None:
            logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
            logs.create_log(model, epoch=epoch, optimizer=optimizer)

def freeze_lower_layers(model, cur_idx):
    if cur_idx == 0:
        return
    for i, module in enumerate(model.module.encoder[:cur_idx]):
        for param in module.parameters():
            param.requires_grad = False
        for param in model.module.loss.get_params(i):
            param.requires_grad = False
        module.eval()
    
    print('Frozen Parameters {}'.format([name for name, param in model.named_parameters() if not param.requires_grad]))
    return

def greedy_train(opt, model, train_loader, optimizer, logs):
    total_step = len(train_loader)

    print_idx = 375
    assert opt.module_layer == -1, 'during training, must account all layers in each block'

    starttime = time.time()

    logs.create_log(model, epoch=-1, optimizer=optimizer)
    if opt.custom_greedy_ep:
        ep_list = np.array([20, 170, 470, 770, 1070, 1370]) 
    else:
        ep_per_idx = (opt.num_epochs + opt.start_epoch)//opt.model_splits
        ep_list = np.arange(1, opt.model_splits + 1) * ep_per_idx
        #ep_list = np.arange(opt.start_epoch, opt.num_epochs + opt.start_epoch, ep_per_idx)
    
    opt.greedy_ep_list = ep_list.tolist()
    print('list of epoch to begin training for each layer {}'.format(ep_list))
    
    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):
        cur_train_module = np.argwhere(ep_list > epoch)[0].item() + 1
        if epoch == ep_list[cur_train_module-1]:
            print('Freeze Layers lower than {} at epoch {}'.format(cur_train_module, epoch))
            freeze_lower_layers(model, cur_train_module)
            

        if opt.distr_strategy == 'ddp':
            train_loader.sampler.set_epoch(epoch)

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_neg_epoch = [0 for i in range(opt.model_splits)]
        loss_pos_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]
        
        # loop over batches in train_loader
        step_start_time = time.time()
        cum_data_time = step_start_time - step_start_time
        cum_it_time = cum_data_time
        for step, (batch_img, label) in enumerate(train_loader):
            
            cum_data_time = cum_data_time + time.time() - step_start_time
            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

                print(
                    "Epoch [{}/{}], Step [{}/{}], Data Time (s): {:.1f}, Computation Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cum_data_time,
                        cum_it_time,
                    )
                )

                starttime = time.time()

            it_start_time = time.time()

            model_input = [img.to(opt.device) for img in batch_img]
            label = label.to(opt.device)

            # forward pass through whole model (loop over modules within model forward)
            loss_dict, h, accuracies = model(model_input, label, n=cur_train_module)

            loss = torch.mean(loss_dict['loss'], 0) # take mean over outputs of different GPUs
            if opt.log_pos_neg:
                loss_pos = torch.mean(loss_dict['loss_pos'], 0) # take mean over outputs of different GPUs
                loss_neg = torch.mean(loss_dict['loss_neg'], 0) # take mean over outputs of different GPUs
            #loss_aux = torch.mean(loss_aux, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracies, 0)

            wandb_log = {'epoch': epoch, 'step': step}
            model.zero_grad()
            try:
                loss[cur_train_module-1].backward()
                optimizer.step()
            except:
                print('loss {}'.format(loss))
                print('cur_train_module {}'.format(cur_train_module))
                raise ValueError('error with greedy training on loss')

            for idx in range(len(loss)):

                # We still output normal (ungated) loss for printing and plotting
                print_loss = loss[idx].item()

                if opt.all_layer and not(opt.pool_layer_loss) and idx > 0:
                    print_loss = print_loss/len(model.module.encoder[idx].model)
                # elif model.module.encoder[idx].use_loss_g:
                #     print_loss *= 0.5

                print_acc = accuracy[idx].item()

                if opt.reg_loss is not None:
                    raise NotImplementedError
                    # print_reg_loss = loss_aux[idx].item()
                    # loss_aux_epoch[idx] += print_reg_loss
                    # print_loss = print_loss - print_reg_loss
                    # wandb_log['reg_loss_{}'.format(idx)] = print_reg_loss 
                if opt.log_pos_neg:
                    print_neg_loss = loss_neg[idx].item()
                    print_pos_loss = loss_pos[idx].item()
                    loss_neg_epoch[idx] += print_neg_loss
                    loss_pos_epoch[idx] += print_pos_loss
                    wandb_log['pos_loss_{}'.format(idx)] = print_pos_loss 
                    wandb_log['neg_loss_{}'.format(idx)] = print_neg_loss 
                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1
                wandb_log['loss_{}'.format(idx)] = print_loss
                wandb_log['acc_{}'.format(idx)] = print_acc
                wandb_log['lr_{}'.format(idx)] = optimizer.param_groups[0]['lr']

                if step % print_idx == 0:
                    print("\t \t Loss_{}: \t \t {:.4f}".format(idx, print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))
            
            for idx in range(len(loss), opt.model_splits):
                loss_epoch[idx] += 0
                loss_updates[idx] += 1
                wandb_log['loss_{}'.format(idx)] = 0
                wandb_log['acc_{}'.format(idx)] = 0
                if opt.log_pos_neg:
                    wandb_log['pos_loss_{}'.format(idx)] = 0
                    wandb_log['neg_loss_{}'.format(idx)] = 0
            
            if WANDB and (opt.distr_strategy != 'ddp' or opt.device_rank == 0):
                wandb.log(wandb_log)

            cum_it_time = cum_it_time + time.time() - it_start_time
            step_start_time = time.time()


        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer, log_module_idx=cur_train_module-1)
    logs.create_log(model, epoch=epoch, optimizer=optimizer)




def main(opt):
     # load model
    model, optimizer = load_vision_model.load_ssl_model_and_optimizer(opt)
    print(model)
    print('Number of Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Trainable Parameters : {}'.format(optim_params))

    logs = logger.Logger(opt)

    utils.seed_everything(opt.seed)
    

    if opt.loss == 1 or opt.use_labeled_train:
        _, _, train_loader, _, _, _ = get_dataloader.get_paried_dataloader(
            opt
        )
        print('Using Labeled Training Dataset!')
    else:
        train_loader, _, _, _, _, _ = get_dataloader.get_paried_dataloader(
            opt
        )
        print('Using Unlabeled Training Dataset!')

    try:
        # Train the model
        if opt.greedy:
            greedy_train(opt, model, train_loader, optimizer, logs)
        else:
            train(opt, model, train_loader, optimizer, logs)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    # logs.create_log(model)

    return


def ddp_main(rank, world_size, opt):

    if WANDB and rank==0:
        wandb.login()

        wandb.init(
            # set the wandb project where this run will be logged
            project="Binary-CL",
            name=opt.save_dir,
            # track hyperparameters and run metadata
            config=vars(opt)
        )

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '2710'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


    model, optimizer = load_vision_model.load_ssl_model_and_optimizer(opt)
    model = DDP(model.to(rank), device_ids=[rank], output_device=rank)
    opt.device_rank = rank
    opt.world_size = world_size
    opt.device = model.device

    print(model)
    print('Number of Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    
    optim_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Trainable Parameters : {}'.format(optim_params))

    print('rank {}, model device {}'.format(rank, model.device))


    logs = logger.Logger(opt) if rank == 0 else None


    utils.seed_everything(opt.seed)
    train_loader, _, supervised_loader, _, _, _ = get_dataloader.get_paried_dataloader(
        opt
    )

    if opt.loss == 1 or opt.use_labeled_train:
        train_loader = supervised_loader
        print('Using Labeled Training Dataset!')

    try:
        # Train the model
        if opt.greedy:
            greedy_train(opt, model, train_loader, optimizer, logs)
        else:
            train(opt, model, train_loader, optimizer, logs)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    # if rank==0:
    #     logs.create_log(model)

    if WANDB and rank==0:
        wandb.finish()

    return





if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.grayscale = False
    # with open('configs/train_vgg.json', 'r') as f:
    #     config = json.load(f)
    # opt.config = config[opt.dataset]

    
    # random seeds
    utils.seed_everything(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    if opt.distr_strategy == 'ddp':
        print('Assuming 1 node being used')
        try:
            ngpus = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size
            mp.spawn(
                ddp_main,
                nprocs=ngpus,
                args=(ngpus, opt),
            )
        except KeyboardInterrupt:
            dist.destroy_process_group()
            raise KeyboardInterrupt("Training got interrupted, saving log-files now.")


        dist.destroy_process_group()

    else:
        if WANDB:
            wandb.login()

            wandb.init(
                # set the wandb project where this run will be logged
                project="Binary-CL",
                name=opt.save_dir,
                # track hyperparameters and run metadata
                config=vars(opt)
            )

        main(opt)
    
        if WANDB:
            wandb.finish()
