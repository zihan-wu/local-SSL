def parse_general_args(parser):
    parser.add_option(
        "--experiment",
        type="string",
        default="vision",
        help="not a real option, just for bookkeeping",
    )
    parser.add_option(
        "--dataset",
        type="string",
        default="stl10",
        help="Dataset to use for training, default: stl10", # cifar10, cifar100
    )
    parser.add_option(
        "--download_dataset",
        action="store_true",
        default=False,
        help="Boolean to decide whether to download the dataset to train on (only tested for STL-10)",
    )
    parser.add_option(
        "--num_epochs", type="int", default=100, help="Number of Epochs for Training"
    )
    parser.add_option("--seed", type="int", default=2, help="Random seed for training")
    parser.add_option("--batch_size", type="int", default=32, help="Batchsize")
    parser.add_option(
        "-i",
        "--data_input_dir",
        type="string",
        default="/lcncluster/zihan/pub-illing2021-neurips/vision/datasets",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    parser.add_option(
        "-o",
        "--data_output_dir",
        type="string",
        default="./",
        help="Directory to store bigger datafiles (dataset and models)",
    )
    parser.add_option(
        "--validate",
        action="store_true",
        default=False,
        help="Boolean to decide whether to split train dataset into train/val and plot validation loss (True) or combine train+validation set for final testing (False)",
    )
    parser.add_option(
        "--only_valid_aug",
        action="store_true",
        default=False,
        help="Only use validation augmentations",
    )
    parser.add_option(
        "--merge_train_unlabeled",
        action="store_true",
        default=False,
        help="Use both train and unlabled dataset to train",
    )
    parser.add_option(
        "--loss",
        type="int",
        default=0,
        help="Loss function to use for training:"
        "0 - Contrastive loss (CLAPP/CPC)"
        "1 - supervised loss using class labels",
    )
    parser.add_option(
        "--use_labeled_train",
        action="store_true",
        default=False,
        help="Use the labeled training dataset",
    )
    parser.add_option(
        "--train_ds_no_shuffle",
        action="store_true",
        default=False,
        help="Use the labeled training dataset",
    )
    parser.add_option(
        "--model",
        type="string",
        default='vgg',
        help="Encoder Model, supporting vgg, resnet, scff"
    )
    parser.add_option(
        "--ete_training",
        action="store_true",
        default=False,
        help="Use the labeled training dataset",
    )
    parser.add_option(
        "--im_size",
        type="int",
        default=64,
        help="ImageNet image size"
    )
    parser.add_option(
        "--grayscale",
        action="store_true",
        default=False,
        help="Boolean to decide whether to convert images to grayscale (default: false)",
    )
    parser.add_option(
        "--add_noise",
        action="store_true",
        default=False,
        help="Boolean to decide whether to add noise to data (default: true)",
    )
    parser.add_option(
        "--weight_init",
        action="store_true",
        default=False,
        help="Boolean to decide whether to use special weight initialization (delta orthogonal)",
    )
    parser.add_option(
        "--save_dir",
        type="string",
        default="",
        help="If given, uses this string to create directory to save results in "
        "(be careful, this can overwrite previous results); "
        "otherwise saves logs according to time-stamp",
    )
    parser.add_option(
        "--distr_strategy",
        type="string",
        default="dp",
        help="strategy of parallelism",
    )
    parser.add_option(
        "--save_freq",
        type="int",
        default=100,
        help="Frequency to save checkpoints",
    )
    parser.add_option(
        "--no_eval_patch_average",
        action="store_true",
        default=False,
        help="remove average pooling over patches in evaluation (default: False, i.e. use average pooling)",
    )

    parser.add_option(
        "--concat_reps_eval",
        action="store_true",
        default=False,
        help="concatenate representations instead of taking one layer in evaluation (default: False, use module_num for evaluation layer)",
    )

    parser.add_option(
        "--subpool_reps",
        type="int",
        default=0,
        help="pooling size of reps",
    )
    return parser
