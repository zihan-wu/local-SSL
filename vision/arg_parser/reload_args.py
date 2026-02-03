from optparse import OptionGroup

def parser_reload_args(parser):
    group = OptionGroup(parser, "Reloading pretrained model options")

    ### Options to load pretrained models
    group.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start GIM training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to pre-trained model needs to be specified in opt.model_path)",
    )
    group.add_option(
        "--model_path",
        type="string",
        default=".",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    group.add_option(
        "--chkpt_path",
        type="string",
        default="/lcncluster/zihan/pub-illing2021-neurips/vision/logs/HingeLossCPC/",
        help="checkpoint to load the weights from ete trained model, for finetuning",
    )
    group.add_option(
        "--old_model_num",
        type="string",
        default="299",
        help="number to load the old model",
    )
    group.add_option(
        "--load_weight_info",
        type="string",
        default="./configs/ete_to_clappres.json",
        help="json file to indicate how old and new weights are mapped",
    )
    group.add_option(
        "--flexible_reload",
        action="store_true",
        default=False,
        help="flexibly reload the closest checkpoint",
    )
    group.add_option(
        "--partial_fintune",
        action="store_true",
        default=False,
        help="Partially finetune top layers",
    )
    group.add_option(
        "--model_num",
        type="string",
        default="100",
        help="Number of the saved model to be used for training the linear classifier"
        "(loaded using model_path + model_X.ckpt, where X is the model_num passed here)",
    )
    group.add_option(
        "--model_type",
        type="int",
        default=0,
        help="Which type of model to use for training of linear classifier on downstream task:"
        "0 - pretrained CLAPP/GreedyInfoMax/CPC model"
        "1 - randomly initialized model"
        "2 - fully supervised model",
    )
    group.add_option(
        "--module_num",
        type="int",
        default=6,
        help="Module to use for training of linear classifier on downstream task (Using 1-indexing). -1 means direct classification on (flattened) images."
        "if multiple layers are used, use the format 1-2-3"
    )
    group.add_option(
        "--multi_module_num",
        type="string",
        default='',
        help="if multiple layers are used, use the format 1-2-3"
    )
    group.add_option(
        "--module_layer",
        type="int",
        default=-1,
        help="layer number (starting from 1) within the module for training classifier; -1 means using whole module, otherwise needs to be positive",
    )
    group.add_option(
        "--in_channels",
        type=int,
        default=None,
        help="Option to explicitly specify the number of input channels for the linear classifier."
        "If None, the default options for resnet output is taken",
    )
    group.add_option(
        "--mse_decode",
        action="store_true",
        default=False,
        help="use mse loss to decode",
    )
    group.add_option(
        "--save_vars_for_update_calc",
        type=int,
        default=-1,
        help="Save intermediate activation for manual update calculation at given layer (1-6). CAREFUL: This constantly increases model size! Only apply to one update!",
    )
    group.add_option(
        "--load_weights_for_gradient_calc",
        action="store_true",
        default=False,
        help="load in weights for gradient calculation",
    )
    group.add_option(
        "--reload_index_path",
        type="string",
        default=None, #'/lcncluster/zihan/pub-illing2021-neurips/vision/logs/rand_index/'
        help="json file to indicate how old and new weights are mapped",
    )

    group.add_option(
        "--reload_fixation_path",
        type="string",
        default=None, #'/lcncluster/zihan/pub-illing2021-neurips/vision/logs/rand_index/'
        help="json file to indicate how fixation/saccade perform",
    )
    group.add_option(
        "--precompute_reps",
        action="store_true",
        default=False,
        help="precompute reps for downstream training",
    )

    parser.add_option_group(group)
    return parser
