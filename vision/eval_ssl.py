import torch
import numpy as np
import time
import os
import code

from vision.data import get_dataloader
from vision.arg_parser import arg_parser
from vision.models import load_vision_model
from vision.utils import logger, utils




def process_reps(opt, outs):
    
    if len(opt.multi_module_num) == 0:
        z = outs[-1].detach()
        if opt.no_eval_patch_average:
            if opt.customize_loss_pool is not None:
                pool_factor = int(opt.customize_loss_pool)
                pool_module = torch.nn.AvgPool2d(kernel_size=pool_factor, stride=pool_factor, padding=0)
                z = pool_module(z).flatten(start_dim=1)
            else:
                z = z.flatten(start_dim=1)
        else:
            z = torch.mean(z, dim=(2, 3))
    else:
        z_list = [outs[int(idx)-1].detach() for idx in opt.multi_module_num.split('-')]
        if opt.no_eval_patch_average:
            if opt.customize_loss_pool is not None:
                pool_factor = [int(w) for w in opt.customize_loss_pool.split('-')]
                mean_pool_modules = [torch.nn.AvgPool2d(kernel_size=pool_factor[int(idx)-1], stride=pool_factor[int(idx)-1], padding=0) for idx in opt.multi_module_num.split('-')]
                z = torch.cat([mean_(z_).flatten(start_dim=1) for z_, mean_ in zip(z_list, mean_pool_modules)], dim=1)
                # mean_pool_modules = [torch.nn.AvgPool2d(kernel_size=pool_factor[int(idx)-1], stride=pool_factor[int(idx)-1], padding=0) for idx in opt.multi_module_num.split('-')]
                # max_pool_modules = [torch.nn.MaxPool2d(kernel_size=pool_factor[int(idx)-1], stride=pool_factor[int(idx)-1], padding=0) for idx in opt.multi_module_num.split('-')]
                # z = torch.cat([mean_(z_).flatten(start_dim=1) + max_(z_).flatten(start_dim=1) for z_, mean_, max_ in zip(z_list, mean_pool_modules, max_pool_modules)], dim=1)
            else:
                z = torch.cat([z_.flatten(start_dim=1) for z_ in z_list], dim=1)
        elif opt.subpool_reps > 0:
            pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding = 0)
            z = torch.cat([pool(z_).flatten(start_dim=1) for z_ in z_list], dim=1)
        else:
            z = torch.cat([torch.mean(z_, dim=(2, 3)) for z_ in z_list], dim=1)

    return z
    


def precompute_embed(opt, context_model, train_loader):
    print('Precomputing embeddings')
    embeds = []
    targets = []
    for step, (img, target) in enumerate(train_loader):
        model_input = img.to(opt.device)

        with torch.no_grad():
            _, outs, _ = context_model(model_input, target, n=opt.module_num, eval=True, module_layer = opt.module_layer)
        z = process_reps(opt, outs) # double security that no gradients go to representation learning part of model
    
        embeds.append(z.cpu())
        targets.append(target)

    return torch.vstack(embeds), torch.concat(targets)



def train_logistic_regression(opt, context_model, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    starttime = time.time()
    val_accus = []

    if opt.use_scheduler:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=opt.num_epochs)

    # No randomness in cifar training -> save reps and go without forward through context model!
    if opt.precompute_reps:
        embeds, targets = precompute_embed(opt, context_model, train_loader)
        print('Embedding shapes {}'.format(embeds.shape))

    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0
        classification_model.train()
        for step, (img, target) in enumerate(train_loader):
            
            classification_model.zero_grad()
            
            
            if opt.precompute_reps:
                start_ind = step*opt.batch_size_multiGPU
                z = embeds[start_ind:(start_ind + len(img))].to(opt.device)
                target = targets[start_ind:(start_ind + len(img))].to(opt.device)
            else:
                model_input = img.to(opt.device)
                #print('model input shape {}, mean {}, std {}'.format(model_input.shape, model_input.mean(), model_input.std()))
                
                with torch.no_grad():
                    _, outs, _ = context_model(model_input, target, n=opt.module_num, eval=True, module_layer = opt.module_layer)
                if step == 0 and epoch == 0:
                    print(len(opt.multi_module_num), [h.shape for h in outs])
                z = process_reps(opt, outs) # double security that no gradients go to representation learning part of model

            prediction = classification_model(z)
            #print('z shape {}, pred shape {}'.format(z.shape, prediction.shape))
            target = target.to(opt.device)
            if opt.mse_decode:
                target_onehot = torch.nn.functional.one_hot(target, classification_model.get_n_class()).float().to(opt.device)
                loss = criterion(prediction, target_onehot)
            else:
                loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if step % 500 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc5,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _ , val_loss = test_logistic_regression(
                opt, context_model, classification_model, test_loader
            )
            logs.append_val_loss([val_loss])
            val_accus.append(val_acc1)
        
        if opt.use_scheduler:
            scheduler.step()
            print("Learning rate: ", optimizer.param_groups[0]['lr'])

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)
        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(
            context_model,
            epoch=epoch,
            classification_model=classification_model,
            accuracy=epoch_acc1 / total_step,
            acc5=epoch_acc5 / total_step,
        )
    if len(val_accus) > 0:
        print("Best validation accuracy: ", max(val_accus))
        return max(val_accus)
    else:
        print("No validation accuracy available, training finished")
        return None
    
        

def test_logistic_regression(opt, context_model, classification_model, test_loader):
    total_step = len(test_loader)
    context_model.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0
    acc1_list = []
    target_list = []

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)
        #print('model input shape {}, mean {}, std {}'.format(model_input.shape, model_input.mean(), model_input.std()))

        with torch.no_grad():
            _, outs, _ = context_model(model_input, target, n=opt.module_num, eval=True, module_layer = opt.module_layer)

        z = process_reps(opt, outs)

        prediction = classification_model(z)

        target = target.to(opt.device)
        if opt.mse_decode:
            target_onehot = torch.nn.functional.one_hot(target, classification_model.get_n_class()).float().to(opt.device)
            loss = criterion(prediction, target_onehot)
        else:
            loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss


        if step % 50 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step

def freeze_layers(model, encoder_idx = [0, 1, 2, 3, 4]):
    for ind in encoder_idx:
        for param in model.module.encoder[ind].parameters():
            param.requires_grad = False
        print('parameters of encoder {} is frozen'.format(ind))
    return model


if __name__ == "__main__":

    opt = arg_parser.parse_args()
    print('opt {}'.format(opt))
    opt.distr_strategy = 'dp'
    #opt.no_eval_patch_average = True
    print('MODIFIED FB Code')

    add_path_var = "linear_regression_model" if opt.mse_decode else"linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # random seeds
    utils.seed_everything(opt.seed)

    # load pretrained model
    # cannot switch opt.reduced_patch_pooling = False here because otherwise W_preds sizes don't match
    context_model, _ = load_vision_model.load_ssl_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )
    print(context_model)
    
    # with torch.no_grad():
    #     _, _, _, z, _ = context_model(0.5*torch.ones((32, 1, 64, 64)), torch.ones(64, dtype=torch.long), n=opt.module_num, eval=True, module_layer = opt.module_layer)
    #     torch.save(z, 'saved_data/ete_250ep.pt')
    #     raise ValueError('tensor saved')
    # model_type=2 is supervised model which trains entire architecture; otherwise just extract features
    if opt.model_type != 2:
        context_model.eval()
        
    if opt.module_num==-1:
        print("CAREFUL! Training classifier directly on input image! Model is ignored and returns the (flattened) input images!")


    classification_model = load_vision_model.load_ssl_classification_model(opt)

    utils.seed_everything(opt.seed)
    _, _, train_loader, _, test_loader, _ = get_dataloader.get_paried_dataloader(opt)

    #context_model = freeze_layers(context_model)

    if opt.model_type == 2:
        if opt.partial_fintune:
            context_model = freeze_layers(context_model)
        params = list(context_model.parameters()) + list(classification_model.parameters())
    else:
        params = list(classification_model.parameters())

    optimizer = torch.optim.Adam(params, lr=0.002 if opt.use_scheduler else 0.001, weight_decay=0 if opt.dataset == 'stl10' else opt.weight_decay) # stl10 is small dataset, so no weight decay

    if opt.mse_decode:
        criterion = torch.nn.MSELoss()
        print('Warning! MSE Loss is used. This should only serve the purpose of a theoretical analysis')
    else:
        criterion = torch.nn.CrossEntropyLoss()
    logs = logger.Logger(opt, pretrain=False)
    print('Number of Trainable Parameters : {}'.format(sum(p.numel() for p in params if p.requires_grad))) 

    try:
        # Train the model
        acc_val = train_logistic_regression(opt, context_model, classification_model, train_loader)

        # Test the model
        acc1, acc5, _ = test_logistic_regression(
            opt, context_model, classification_model, test_loader
        )

    except KeyboardInterrupt:
        print("Training got interrupted")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1,
        acc5=acc5,
        final_test=True,
    )
    # torch.save(
    #     context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    # )
    L = ["Test top1 classification accuracy: "+str(acc1)+"\n",
        "Test top5 classification accuracy: "+str(acc5)+"\n"]
    if acc_val is not None:
        L.append("Validation top1 classification accuracy: "+str(acc_val)+"\n")

    classify_type = "flatten_classification" if opt.no_eval_patch_average else 'classification{}'.format('' if opt.subpool_reps == 0 else opt.subpool_reps)
    
    np.save(os.path.join(opt.model_path, "{}_{}ep_{}_values_".format(classify_type, opt.model_num, opt.module_num if len(opt.multi_module_num) == 0 else opt.multi_module_num)+str(opt.dataset)+"_aug.npy"), 
            np.array([acc1, acc5]))
    
    f = open(os.path.join(opt.model_path, "{}_{}ep_{}_".format(classify_type, opt.model_num, opt.module_num if len(opt.multi_module_num) == 0 else opt.multi_module_num)+str(opt.dataset)+"_aug.txt"), "w")
    f.writelines(L)
    f.close()
