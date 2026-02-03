import torch

from vision.models import FullModel, ClassificationModel, SSLModel
from vision.utils import model_utils, opt_scheduler


def load_model_and_optimizer(opt, num_GPU=None, reload_model=False, calc_loss=True):

    model = FullModel.FullVisionModel(
        opt, calc_loss
    )

    optimizer = []
    if opt.model_splits == 1:
        optimizer.append(torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay))
    elif opt.model_splits >= 2:
        # use separate optimizer for each module, so gradients don't get mixed up
        for idx, layer in enumerate(model.encoder):
            optimizer.append(torch.optim.Adam(layer.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay))
    else:
        raise NotImplementedError
    # Note: module.parameters() acts recursively by default and adds all parameters of submodules as well

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model, optimizer = model_utils.reload_weights(
        opt, model, optimizer, reload_model=reload_model
    )

    return model, optimizer


def load_ssl_model_and_optimizer(opt, num_GPU=None, reload_model=False, calc_loss=True):

    model = SSLModel.ContrastiveVision(
        opt, calc_loss
    )

    
    #optimizer = []
    if opt.dataset == 'imagenet':
        print("Using LARS optimizer")
        param_groups = [{
            'params': [p for name, p in model.named_parameters()],
            'weight_decay': opt.weight_decay,
            'layer_adaptation': True,
        }]
        optimizer = torch.optim.SGD(param_groups, lr=opt.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # Note: module.parameters() acts recursively by default and adds all parameters of submodules as well

    model, optimizer = model_utils.reload_ssl_weights(
        opt, model, optimizer, reload_model=reload_model, calc_loss=calc_loss
    )

    if opt.distr_strategy == 'dp':
        model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)
    

    return model, optimizer


def load_ssl_model_and_optimizer_new(opt, num_GPU=None, reload_model=False, calc_loss=True):

    model = SSLModel.ContrastiveVision(
        opt, calc_loss
    )

    
    optimizer = []
    if calc_loss:
        if opt.model_splits == 1:
            optimizer.append(torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay))
        elif opt.model_splits >= 2:
            # use separate optimizer for each module, so gradients don't get mixed up
            for idx, layer in enumerate(model.encoder):
                #optimizer.append(torch.optim.Adam(list(layer.parameters()) + list(model.loss.get_params(idx)), lr=opt.learning_rate, weight_decay=opt.weight_decay))
                optimizer.append(torch.optim.AdamW(list(layer.parameters()) + list(model.loss.get_params(idx)), lr=opt.config['opt_configs'][idx]['lr'], weight_decay=opt.config['opt_configs'][idx]['weight_decay']))
                print([p.shape for p in optimizer[-1].param_groups[0]['params']])
        else:
            raise NotImplementedError
    # optimizer=torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # Note: module.parameters() acts recursively by default and adds all parameters of submodules as well

    

    model, optimizer = model_utils.reload_ssl_weights(
        opt, model, optimizer, reload_model=reload_model, calc_loss=calc_loss
    )

    if opt.distr_strategy == 'dp':
        model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)
    

    return model, optimizer


def load_classification_model(opt):
    if opt.in_channels == None:
        in_channels = 1024
    else:
        in_channels = opt.in_channels

    if opt.dataset == "stl10" or opt.dataset == "cifar10" or opt.dataset == "mnist":
        num_classes = 10
    elif opt.dataset == "cifar100":
        num_classes = 100
    elif opt.dataset == "imagenet":
        num_classes = 1000
    else:
        raise Exception("Invalid option")

    if opt.no_eval_patch_average:
        classification_model = ClassificationModel.FlattenClassificationModel(
            in_channels=in_channels, num_classes=num_classes,
        ).to(opt.device)

    else:
        classification_model = ClassificationModel.ClassificationModel(
            in_channels=in_channels, num_classes=num_classes,
        ).to(opt.device)

    return classification_model


def load_ssl_classification_model(opt):
    if opt.in_channels == None:
        in_channels = 1024
    else:
        in_channels = opt.in_channels

    if opt.dataset == "stl10" or opt.dataset == "cifar10" or opt.dataset == "mnist":
        num_classes = 10
    elif opt.dataset == "cifar100":
        num_classes = 100
    elif opt.dataset == "imagenet":
        num_classes = 1000
    elif opt.dataset == "tiny_imagenet":
        num_classes = 200
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.FlattenClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model

