import torch
import torch.nn as nn
import os
import glob
import re
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler

        

def distribute_over_GPUs(opt, model, num_GPU):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU




def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)


def reload_weights(opt, model, optimizer, reload_model):
    ## reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                if idx < opt.train_module:
                    if opt.flexible_reload:
                        files=glob.glob(os.path.join(
                                            opt.model_path,
                                            "model_{}_*.ckpt".format(idx)))
                        available_ckpt = np.sort(np.array([int(re.findall(r"[-+]?\d+", f)[-1]) for f in files]))
                        actual_model_num = np.extract(available_ckpt <= int(opt.model_num), available_ckpt)[-1] 
                    else:
                        actual_model_num = opt.model_num

                    print('weights from layer {} , module number {} are loaded'.format(idx, actual_model_num))
                    model.module.encoder[idx].load_state_dict(
                        torch.load(
                            os.path.join(
                                opt.model_path,
                                "model_{}_{}.ckpt".format(idx, actual_model_num),
                            ),
                            map_location=opt.device.type,
                        )
                    )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )
            

        for i, optim in enumerate(optimizer):
            fname = os.path.join(
                        opt.model_path,
                        "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                    )
            if os.path.isfile(fname):
                optim.load_state_dict(
                    torch.load(
                        fname,
                        map_location=opt.device.type,
                    )
                )
                for g in optim.param_groups:
                    g['lr'] = opt.learning_rate
            else:
                print('Warning: {} optimizer path not exist, re-initialize from the start'.format(fname))
                optim = torch.optim.Adam(model.module.encoder[i].parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        print("Randomly initialized model")

    return model, optimizer



def reload_ssl_weights(opt, model, optimizer, reload_model, calc_loss=False):
    ## reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.encoder):
                if idx < opt.train_module:
                    if opt.flexible_reload:
                        files=glob.glob(os.path.join(
                                            opt.model_path,
                                            "model_{}_*.ckpt".format(idx)))
                        available_ckpt = np.sort(np.array([int(re.findall(r"[-+]?\d+", f)[-1]) for f in files]))
                        print('available ckpt to load from model_{}: {}'.format(idx, available_ckpt))
                        actual_model_num = np.extract(available_ckpt <= int(opt.model_num), available_ckpt)[-1] 
                    else:
                        actual_model_num = opt.model_num

                    print('weights from layer {} , module number {} are loaded'.format(idx, actual_model_num))
                    model.encoder[idx].load_state_dict(
                        torch.load(
                            os.path.join(
                                opt.model_path,
                                "model_{}_{}.ckpt".format(idx, actual_model_num),
                            ),
                            map_location=opt.device.type,
                        )
                    )
            if calc_loss:
                print("Loaded loss weights from ", opt.model_path)
                model.loss.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "loss_{}.ckpt".format(actual_model_num),
                        ),
                        map_location=opt.device.type,
                    )
                )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.encoder):
                model.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )
            print("Loaded weights from ", opt.model_path)
            if calc_loss:
                print("Loaded loss weights from ", opt.model_path)
                model.loss.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "loss_{}.ckpt".format(opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )
        
        if isinstance(optimizer, list):
            for i, optim in enumerate(optimizer):
                fname = os.path.join(
                            opt.model_path,
                            "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                        )
                if os.path.isfile(fname):
                    optim.load_state_dict(
                        torch.load(
                            fname,
                            map_location=opt.device.type,
                        )
                    )
                    for g in optim.param_groups:
                        g['lr'] = opt.learning_rate
                else:
                    print('Warning: {} optimizer path not exist, re-initialize from the start'.format(fname))
                    optim = torch.optim.AdamW(model.encoder[i].parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

        else:
            fname = os.path.join(
                        opt.model_path,
                        "optim_{}.ckpt".format(opt.start_epoch),
                    )
            if os.path.isfile(fname):
                optimizer.load_state_dict(
                    torch.load(
                        fname,
                        map_location=opt.device.type,
                    )
                )
                for g in optimizer.param_groups:
                    g['lr'] = opt.learning_rate
            else:
                print('Warning: {} optimizer path not exist, re-initialize from the start'.format(fname))
                optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        print("Randomly initialized model")

    return model, optimizer




def reload_ssl_weights_dp(opt, model, optimizer, reload_model, calc_loss=False):
    ## reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                if idx < opt.train_module:
                    if opt.flexible_reload:
                        files=glob.glob(os.path.join(
                                            opt.model_path,
                                            "model_{}_*.ckpt".format(idx)))
                        available_ckpt = np.sort(np.array([int(re.findall(r"[-+]?\d+", f)[-1]) for f in files]))
                        actual_model_num = np.extract(available_ckpt <= int(opt.model_num), available_ckpt)[-1] 
                    else:
                        actual_model_num = opt.model_num

                    print('weights from layer {} , module number {} are loaded'.format(idx, actual_model_num))
                    model.module.encoder[idx].load_state_dict(
                        torch.load(
                            os.path.join(
                                opt.model_path,
                                "model_{}_{}.ckpt".format(idx, actual_model_num),
                            ),
                            map_location=opt.device.type,
                        )
                    )
            if calc_loss:
                print("Loaded loss weights from ", opt.model_path)
                model.module.loss.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "loss_{}.ckpt".format(actual_model_num),
                        ),
                        map_location=opt.device.type,
                    )
                )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )
            print("Loaded weights from ", opt.model_path)
            if calc_loss:
                print("Loaded loss weights from ", opt.model_path)
                model.module.loss.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "loss_{}.ckpt".format(actual_model_num),
                        ),
                        map_location=opt.device.type,
                    )
                )
        
        if isinstance(optimizer, list):
            for i, optim in enumerate(optimizer):
                fname = os.path.join(
                            opt.model_path,
                            "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                        )
                if os.path.isfile(fname):
                    optim.load_state_dict(
                        torch.load(
                            fname,
                            map_location=opt.device.type,
                        )
                    )
                    for g in optim.param_groups:
                        g['lr'] = opt.learning_rate
                else:
                    print('Warning: {} optimizer path not exist, re-initialize from the start'.format(fname))
                    optim = torch.optim.AdamW(model.module.encoder[i].parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

        else:
            fname = os.path.join(
                        opt.model_path,
                        "optim_{}.ckpt".format(opt.start_epoch),
                    )
            if os.path.isfile(fname):
                optim.load_state_dict(
                    torch.load(
                        fname,
                        map_location=opt.device.type,
                    )
                )
                for g in optim.param_groups:
                    g['lr'] = opt.learning_rate
            else:
                print('Warning: {} optimizer path not exist, re-initialize from the start'.format(fname))
                optim = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        print("Randomly initialized model")

    return model, optimizer