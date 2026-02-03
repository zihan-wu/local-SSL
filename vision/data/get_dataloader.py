import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os, glob
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from vision.arg_parser import arg_parser
from torchvision.io import read_image, ImageReadMode

DATA_ROOT = '/lcncluster/zihan/MNIST/dataset'
IMAGENET_PATH = '/lcncluster/datasets/ImageNet/root_ImageNet' #'/lcnscratch/datasets/root_ImageNet' # '/lcncluster/datasets/ImageNet/root_ImageNet'

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.filenames = glob.glob("{}/*/*/*.JPEG".format(data_dir))
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

    

class SimCLRTrainDataTransform:
    """Transforms for SimCLR.

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, scale_size=(0.08, 1.0), gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ) -> None:


        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height, scale = scale_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1
            gaussian_blur = transforms.GaussianBlur(kernel_size)
            data_transforms.append(transforms.RandomApply([gaussian_blur], p=0.5))

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

    
    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj


class SimCLREvalDataTransform:
    """Transforms for SimCLR.

    Transform::
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, scale_size=(0.08, 1.0), normalize=None, aug=False
    ):

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        # replace online transform with eval time transform
        if aug:
            self.online_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_height,scale=scale_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    self.final_transform,
                ]
            )
        else:
            self.online_transform = transforms.Compose(
            [
                transforms.CenterCrop(input_height),
                self.final_transform,
            ]
        )

    def __call__(self, sample):

        return self.online_transform(sample)
    


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    
def get_simclr_transform(img_size, normalization, pretrain=True, scale_size=(0.08, 1.0)):
    jitter_strength=1.0
    color_jitter = transforms.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
    )
    kernel_size = int(0.1 * img_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    gaussian_blur = transforms.GaussianBlur(kernel_size)
    if pretrain:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=scale_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([gaussian_blur], p=0.5),
            transforms.ToTensor(),
            # These are normalisation factors found online.
            transforms.Normalize(normalization[0], normalization[1]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size, scale=scale_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # These are normalisation factors found online.
            transforms.Normalize(normalization[0], normalization[1]),
        ])

    test_transform = transforms.Compose([
        transforms.Resize(int(img_size+ 0.1 * img_size)),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        # These are normalisation factors found online.
        transforms.Normalize(normalization[0], normalization[1]),
    ])

    return train_transform, test_transform


def get_paried_dataloader(opt):



    if opt.dataset == "stl10":
        print("load STL-10 dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_paired_dataloader(
            opt
        )
    elif opt.dataset == "cifar10" or opt.dataset == "cifar100":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_cifar_paired_dataloader(
            opt
        )
    elif opt.dataset == "tiny_imagenet":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_timagenet_paired_dataloader(
            opt
        )
    elif opt.dataset == "imagenet":
        print("load ImageNet dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_imagenet_paired_dataloader(
            opt
        )
    else:
        raise Exception("Invalid option")

    # embed()
    # raise Exception()
    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )



def get_cifar_paired_dataloader(opt):
    print('Loading Cifar Dataset ...')
    base_folder = opt.data_input_dir

    norm = transforms.Normalize(mean=[125.3/255, 123.0/255, 113.9/255], std=[63.0/255, 62.1/255, 66.7/255]) if opt.dataset == "cifar10" else transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    train_transform = SimCLRTrainDataTransform(32, scale_size=(0.25, 1.0), normalize=norm)
    test_transform = SimCLREvalDataTransform(32, normalize=norm)
    decode_transform = SimCLREvalDataTransform(32, normalize=norm, aug=True)

    # normalization = ((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))
    # train_transform, test_transform = get_simclr_transform(32, normalization)

    

    if opt.only_valid_aug:
        print('ONLY using augmentations for validations')
        train_transform = decode_transform 

    ds_class = torchvision.datasets.CIFAR10 if opt.dataset == "cifar10" else torchvision.datasets.CIFAR100
    
    unsupervised_dataset = ds_class(
        base_folder,
        train=True,
        transform=train_transform, #ContrastiveTransformations(train_transform), #
        download=opt.download_dataset,
    ) #set download to True to get the dataset

    train_dataset = ds_class(
        base_folder, train=True, transform=decode_transform, download=opt.download_dataset
    )

    test_dataset = ds_class(
        base_folder, train=False, transform=test_transform, download=opt.download_dataset
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(unsupervised_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, 
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def get_imagenet_paired_dataloader(opt):
    print('Loading ImageNet Dataset ...')
    base_folder = opt.data_input_dir
    # normalization = ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))

    # train_transform, test_transform = get_simclr_transform(64, normalization)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) #* 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) #* 255
    transform_train = SimCLRTrainDataTransform(224, normalize=transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)) #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    # transform_decode = SimCLREvalDataTransform(224, normalize=transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), aug=True)
    # transform_test = SimCLREvalDataTransform(224, normalize=transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), aug=False)
    transform_decode, transform_test = get_simclr_transform(224, (IMAGENET_MEAN, IMAGENET_STD), pretrain=False)

    if opt.only_valid_aug:
        print('ONLY using augmentations for validations')
        transform_train = transform_decode

    unsupervised_dataset = torchvision.datasets.ImageNet(
        base_folder,
        split='train',
        #os.path.join(base_folder, 'train'),
        transform=transform_train, #transform_train,
    ) #set download to True to get the dataset

    # unsupervised_dataset = torch.utils.data.Subset(unsupervised_dataset, range(10000))


    train_dataset = torchvision.datasets.ImageNet(
        base_folder,
        split='train',
        #os.path.join(base_folder, 'train'),
        transform=transform_decode
    )

    test_dataset = torchvision.datasets.ImageNet(
        base_folder,
        split='val',
        #os.path.join(base_folder, 'val'),
        transform=transform_test
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=8, pin_memory=True, persistent_workers=True, 
        sampler=DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(unsupervised_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True, 
        sampler=DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def get_timagenet_paired_dataloader(opt):
    print('Loading TinyImageNet Dataset ...')
    base_folder = opt.data_input_dir
    # normalization = ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))

    # train_transform, test_transform = get_simclr_transform(64, normalization)
    transform_train = SimCLRTrainDataTransform(64, normalize=transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]))
    transform_valid = SimCLREvalDataTransform(64, normalize=transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]), aug=True)
    transform_test = SimCLREvalDataTransform(64, normalize=transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]))

    if opt.only_valid_aug:
        print('ONLY using augmentations for validations')
        transform_train = transform_valid

    unsupervised_dataset = torchvision.datasets.ImageFolder(
        os.path.join(base_folder, 'train'),
        transform=transform_train, #transform_train,
    ) #set download to True to get the dataset


    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(base_folder, 'train'),
        transform=transform_valid
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(base_folder, 'val'),
        transform=transform_test
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(unsupervised_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, 
        num_workers=16, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.device_rank, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def get_stl10_paired_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    transform_train = SimCLRTrainDataTransform(opt.random_crop_size, normalize=transforms.Normalize(mean=[0.4313, 0.4156, 0.3663], std=[0.2683, 0.2610, 0.2687]))
    #transform_valid = SimCLREvalDataTransform(opt.random_crop_size, normalize=transforms.Normalize(mean=[0.4313, 0.4156, 0.3663], std=[0.2683, 0.2610, 0.2687]))
    transform_decode, transform_test = get_simclr_transform(opt.random_crop_size, ([0.4313, 0.4156, 0.3663], [0.2683, 0.2610, 0.2687]), pretrain=False)

    if opt.only_valid_aug:
        print('ONLY using augmentations for validations')
        transform_train = transform_decode

    
    
    unsupervised_split = 'train+unlabeled' if opt.merge_train_unlabeled else 'unlabeled'
    
    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split=unsupervised_split,
        transform=transform_train,
        download=opt.download_dataset,
    ) #set download to True to get the dataset


    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_decode, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_test, download=opt.download_dataset
    )


    
    
    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=8, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(train_dataset, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.train_ds_no_shuffle) and (opt.distr_strategy != 'ddp'),
        num_workers=8, pin_memory=True, persistent_workers=True,
        sampler=DistributedSampler(unsupervised_dataset, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )



    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=8,
        sampler=DistributedSampler(test_dataset, shuffle=(not opt.train_ds_no_shuffle)) if opt.distr_strategy=='ddp' else None
    )

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )



def get_patch_dataloader(opt):
    if opt.dataset == "stl10":
        print("load STL-10 dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
    elif opt.dataset == "cifar10" or opt.dataset == "cifar100":
        print("load CIFAR dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_cifar_dataloader(
            opt
        )
        # train_loader and train_dataset are None in this case!
    elif opt.dataset == "mnist":
        print("load MNIST dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_mnist_dataloader(
            opt
        )
    elif opt.dataset == "imagenet":
        print("load ImageNet dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_imagenet_dataloader(
            opt
        )
    else:
        raise Exception("Invalid option")

    # embed()
    # raise Exception()
    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )

def get_mnist_dataloader(opt):
    mnist_transform = transforms.Compose([
        #transforms.CenterCrop(24),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root=DATA_ROOT, download=True, transform=mnist_transform)
    test_dataset = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=mnist_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )
    return (
        train_loader,
        train_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def get_stl10_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
            "randcrop": opt.random_crop_size,
            "flip": True,
            "resize": False,
            "pad": False,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    if opt.add_noise:
        transform_train = transforms.Compose([transform_train, AddGaussianNoise(0, 0.25)])
        transform_valid = transforms.Compose([transform_valid, AddGaussianNoise(0, 0.25)])
        print('NOISE Added to Decoding Performance')

    if opt.only_valid_aug:
        print('ONLY using augmentations for validations')
        transform_train = transform_valid

    
    
    unsupervised_split = 'train+unlabeled' if opt.merge_train_unlabeled else 'unlabeled'
    
    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split=unsupervised_split,
        transform=transform_train,
        download=opt.download_dataset,
    ) #set download to True to get the dataset


    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    )

    
    
    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=(not opt.train_ds_no_shuffle), num_workers=16
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=(not opt.train_ds_no_shuffle),
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        if opt.training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        elif opt.training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        else:
            raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            base_folder,
            split=opt.training_dataset,
            transform=transform_valid,
            download=opt.download_dataset,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=valid_sampler,
            num_workers=16,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )

def get_imagenet_dataloader(opt):
    im_size = opt.im_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            im_size,
            scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        Clip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        CenterCropAndResize(proportion=0.875, size=im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    traindir = os.path.join(IMAGENET_PATH, 'train')
    testdir = os.path.join(IMAGENET_PATH, 'val')

    unsupervised_set = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
    #unsupervised_set = torch.utils.data.Subset(unsupervised_set, range(64))
    trainset = torchvision.datasets.ImageFolder(traindir, transform=train_transform)
    #trainset = torch.utils.data.Subset(trainset, range(64))
    testset = torchvision.datasets.ImageFolder(testdir, transform=test_transform)
    print('Length of Train and Val dataset {}, {}'.format(len(trainset), len(testset)))


    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )
    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_set, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )

    return (
        unsupervised_loader,
        unsupervised_set,
        train_loader,
        trainset,
        test_loader,
        testset,
    )




class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)
    
class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped and image.
        """
        w, h = (np.array(img.size) * self.proportion).astype(int)
        img = torchvision.transforms.functional.resize(
            torchvision.transforms.functional.center_crop(img, (h, w)),
            (self.size, self.size),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)

def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

# only exists in v0.9.9
# class Sharpen:
#     """Sharpen image after upsampling with interpolation."""
#     def __call__(self, x):
#         return TF.adjust_sharpness(x, 2.)

def get_transforms(eval=False, aug=None):
    trans = []

    if aug["resize"]:
        trans.append(transforms.Resize(aug["resize_size"]))

    if aug["pad"]:
        trans.append(transforms.Pad(aug["pad_size"], fill=0, padding_mode='constant'))
    
    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans


def get_cifar_dataloader(opt):
    cor_factor_mean = 0.06912513 # correction factors: STL-10 normalisation lead to these residual mean and std -> has to be adapted to get same input distribution
    cor_factor_std = 0.95930314
    if opt.dataset == "cifar10":
        print("load cifar10 dataset...")
        base_folder = os.path.join(opt.data_input_dir, "cifar10_binary")
        bw_mean = 0.47896898 - cor_factor_mean * 0.2392343 / cor_factor_std
        bw_std = 0.2392343 / cor_factor_std
    elif opt.dataset == "cifar100":
        print("load cifar100 dataset...")
        base_folder = os.path.join(opt.data_input_dir, "cifar100_binary")
        bw_mean = 0.48563015 - cor_factor_mean * 0.25072286 / cor_factor_std
        bw_std = 0.25072286 / cor_factor_std

    aug = {
        "cifar": {
            "resize": False,
            "resize_size": 64, # 96
            "pad": False,
            "pad_size": 16,
            "randcrop": False, #opt.random_crop_size,
            "flip": False,
            "grayscale": opt.grayscale,
            "bw_mean": [bw_mean],
            "bw_std": [bw_std],
        }
    }
    # mean and std found as:
    # x = np.concatenate([np.asarray(im) for (im, t) in supervised_loader]); np.mean(x); np.std(x)
    # CIFAR10
    # for vanilla 32 x 32 input: "bw_mean": [0.47896898], "bw_std": [0.2392343]
    # for resize_size: 96 and randcrop: "bw_mean": [0.470379], "bw_std": [0.2249]
    # for resize_size: 64 without randcrop: "bw_mean": [0.4798809], "bw_std": [0.23278822]
    # for pad: True and pad_size: 16: "bw_mean": [0.11974239], "bw_std": [0.23942184]
    # CIFAR100
    # for vanilla 32 x 32 input: "bw_mean": [0.48563015], "bw_std": [0.25072286]

    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["cifar"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["cifar"])]
    )

    if opt.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            base_folder, train=True, transform=transform_train, download=opt.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR10(
            base_folder, train=False, transform=transform_valid, download=opt.download_dataset
        )
    elif opt.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            base_folder, train=True, transform=transform_train, download=opt.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR100(
            base_folder, train=False, transform=transform_valid, download=opt.download_dataset
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    return (
        None,
        None,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )




if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.batch_size_multiGPU = opt.batch_size
    opt.dataset = 'cifar10'
    opt.data_input_dir = '/Users/zihanwu/Desktop/EPFL/LCN/CIFAR/dataset'
    train_loader, _, supervised_loader, _, test_loader, _ = get_paried_dataloader(opt)
    train1 = next(iter(train_loader))
    eval1 = next(iter(supervised_loader))
    test1 = next(iter(test_loader))
    


