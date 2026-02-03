from optparse import OptionGroup

def parse_train_args(parser):
    group = OptionGroup(parser, "Training options")
    group.add_option(
        "--learning_rate", 
        type="float", 
        default=2e-4, 
        help="Learning rate (for ADAM optimiser)"
    )
    group.add_option(
        "--weight_decay", 
        type="float", 
        default=0., 
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )
    group.add_option(
        "--pos_coeff", 
        type="float", 
        default=1, 
        help="ratio of positive loss to negative loss in the contrastive loss function"
    )
    group.add_option(
        "--a_decay", 
        type="float", 
        default=5e-4, 
        help="decay for just pred projection"
    )
    group.add_option(
        "--l2_penalty",
        action="store_true",
        default=False,
        help="l2 penalty coefficient for pred projections",
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=5,
        help="(Number of) Time steps to predict into future",
    )
    group.add_option(
        "--gradual_prediction_steps",
        action="store_true",
        default=False,
        help="Increase number of time steps (to predict into future) module by module. This is meant to be used with 6 modules",
    )
    group.add_option(
        "--reduced_patch_pooling",
        action="store_true",
        default=False,
        help="Reduce adaptive average pooling of patch encodings. This means that some spatial information is kept." 
            "The dimension of context and target vectors grow accordingly. This is meant to be used with 6 modules.",
    )
    group.add_option(
        "--reduce_patch_factor",
        type="int",
        default=2,
        help="factor to pool spatial info in each patch",
    )
    group.add_option(
        "--remove_patch_pooling",
        action="store_true",
        default=False,
        help="No path spatial pooling, all spatial info stayed, but pooled by the reduce patch factor as specified above",
    )
    group.add_option(
        "--spatial_socre",
        action="store_true",
        default=False,
        help="Use the spatial informtion to compute score u before conducting spatial pooling",
    )
    group.add_option(
        "--proj_2d",
        action="store_true",
        default=False,
        help="use 2d projection",
    )
    group.add_option(
        "--contrain_norm",
        action="store_true",
        default=False,
        help="constrain the norm of W_ff",
    )
    group.add_option(
        "--pred_decay",
        type="float", 
        default=0, 
        help="decay of predictive projection"
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=1,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--current_rep_as_negative",
        action="store_true",
        default=False,
        help="Use the current feature vector ('context' at time t as opposed to predicted time step t+k) itself as/for sampling the negative sample",
    )
    group.add_option(
        "--sample_negs_locally",
        action="store_true",
        default=False,
        help="Sample neg. samples from batch but within same location in image, i.e. no shuffling across locations",
    )
    group.add_option(
        "--sample_negs_locally_same_everywhere",
        action="store_true",
        default=False,
        help="Extension of --sample_negs_locally_same_everywhere (must be True). No shuffling across locations and same sample (from batch) for all locations. I.e. negative sample is simply a new input without any scrambling",
    )
    group.add_option(
        "--avoid_same_neg_sample",
        action="store_true",
        default=False,
        help="Avoid samlping the same image for negative contrast",
    )
    group.add_option(
        "--detach_separately",
        action="store_true",
        default=False,
        help="detach loss twice, W_pred optimized twice",
    )
    group.add_option(
        "--either_pos_or_neg_update",
        action="store_true",
        default=False,
        help="Randomly chose to do either pos or neg update in Hinge loss. --negative_samples should be 1. Only used with --current_rep_as_negative True",
    )
    group.add_option(
        "--log_pos_neg",
        action="store_true",
        default=False,
        help="Log both Positive and Negative loss. Warn: Only when 'which update' is both",
    )
    group.add_option(
        "--patch_size",
        type="int",
        default=16,
        help="Encoding patch size. Use single integer for same encoding size for all modules (default=16)",
    )
    group.add_option(
        "--increasing_patch_size",
        action="store_true",
        default=False,
        help="Boolean: start with patch size 4 and increase by factors 2 per module until max. patch size = --patch_size (e.g. 16)",
    )
    group.add_option(
        "--random_crop_size",
        type="int",
        default=64,
        help="Size of the random crop window. Use single integer for same size for all modules (default=64) For CIFAR, Tiny-ImageNet, ImageNet, they are fixed to 32, 64, 224 respectively",
    )
    group.add_option(
        "--inpatch_prediction",
        action="store_true",
        default=False,
        help="Boolean: change CPC task to smaller scale prediction (within patch -> smaller receptive field) by extra unfolding ",
    )
    group.add_option(
        "--inpatch_prediction_limit",
        type="int",
        default=2,
        help="Number of module below which inpatch prediction is applied (if inpatch prediction is active) (default=2, i.e. modules 0 and 1 are doing inpatch prediction)",
    )
    group.add_option(
        "--feedback_gating",
        action="store_true",
        default=False,
        help="Boolean: use feedback from higher layers to gate lower layer plasticity",
    )
    group.add_option(
        "--gating_av_over_preds",
        action="store_true",
        default=False,
        help="Boolean: average feedback gating (--feedback_gating) from higher layers over different prediction steps ('k')",
    )
    group.add_option(
        "--contrast_mode",
        type="str",
        default="hinge",
        help="decreasing convex function f, type 2 choices are 'hinge', 'linear', 'softhinge' (i.e. softplus), or type 1 'phyll'; also 'infonce' for InfoNCE",
    )
    group.add_option(
        "--detach_c",
        action="store_true",
        default=False,
        help="Boolean whether the gradient of the context c should be dropped (detached)",
    )
    group.add_option(
        "--encoder_type",
        type="str",
        default="vgg_like",
        help="Select the encoder type: resnet or vgg_like",
    )

    group.add_option(
        "--custom_mlp",
        type="str",
        default=None,
        help="index of blocks to replace cnn with mlp, splitted with'-': particularly for the final layers whether further convolution does not make sense "
    )
    
    group.add_option(
        "--input_decorr_layer",
        type="str",
        default=None,
        help="index of blocks to augment cnn with decorrelated input, splitted with'-': particularly for the final layers whether further convolution does not make sense "
    )

    group.add_option(
        "--inference_recurrence",
        type="int",
        default=0,
        help="recurrence (on the module level) during inference (before evaluating loss):"
        "0 - no recurrence"
        "1 - lateral recurrence within layer"
        "2 - feedback recurrence"
        "3 - both, lateral and feedback recurrence",
    )
    group.add_option(
        "--recurrence_iters",
        type="int",
        default=5,
        help="number of iterations for inference recurrence (without recurrence, --inference_recurrence == 0, it is set to 0) ",
    )
    group.add_option(
        "--model_splits",
        type="int",
        default=6,
        help="Number of individually trained modules that the original model should be split into "
             "options: 1 (normal end-to-end backprop) or 3 (default used in experiments of paper)",
    )
    group.add_option(
        "--ete_n_block",
        type="int",
        default=6,
        help="number of layers to do ete training",
    )
    group.add_option(
        "--only_first_layers",
        action="store_true",
        default=False,
        help="if true, model splits < 6 means using earlier layer",
    )
    group.add_option(
        "--resnet_splits",
        type="string",
        default='blockwise',
        help="Level of training in resnet: layerwise, blockwise, subblockwise"
    )
    group.add_option(
        "--resnet_num",
        type="int",
        default=18,
        help="number of resnet layers to use, 18, 34, 50, 101, 152",
    )
    group.add_option(
        "--train_module",
        type="int",
        default=6,
        help="Index of the module to be trained individually (0-2), "
        "or training network as one (3)",
    )
    group.add_option(
        "--predict_module_num",
        type="str",
        default="same",
        help="Option whether W should predict activities in the same module ('same', default), "
             "one module below with first module predicting same module ('-1'),"
             "both ('both') or"
             "one module below with last module predicting same module ('-1b')"
             "both for -1b ('bothb')"
             "both but in the sense of combine same layer and layer above in one loss, i.e. same saccade/fixation ('-1c')",
    )
    group.add_option(
        "--customize_fb_idx",
        type="str",
        default=None,
        help="str of customized fb_idx, must follow the format 2-2-4-4-6-6 or 4-5-6-4-5-6"
    )
    group.add_option(
        "--dfa",
        action="store_true",
        default=False,
        help="direct feedback alignment, only useful when pred_module is fb",
    )
    group.add_option(
        "--train_wm",
        action="store_true",
        default=False,
        help="train dfa with weighted mirror type",
    )
    group.add_option(
        "--reg_loss",
        type="str",
        default=None,
        help="Option whether regularization term is included in loss "
             "default ('') is not using regularization"
             "decorr is for decorrelation term in Vicreg",
    )

    group.add_option(
        "--decode_loss",
        type="str",
        default=None,
        help="Option whether decode loss is included in loss "
             "default ('') is not using decode loss"
             "decode is for reconstruct layer by layer",
    )

    group.add_option(
        "--decode_loss_coeff",
        type="float",
        default=1.0,
        help="Coefficient for the decode loss",
    )
    group.add_option(
        "--reg_block_idx",
        type="str",
        default='0-6',
        help="range of block index to apply reg_loss, [a, b) written in 'a-b' "
    )
    group.add_option(
        "--detach_upper_layer",
        action="store_true",
        default=False,
        help="detach the upper layer in case of top down connections"
             "only make sense for both(b), -1b, and -1c",
    )
    group.add_option(
        "--extra_conv",
        action="store_true",
        default=False,
        help="Boolian whether extra convolutional layer too increase rec. field size (with downsampling, i.e. stride > 1)"
             "is used to decode activity before avg-pooling and contrastive loss",
    )
    group.add_option(
        "--asymmetric_W_pred",
        action="store_true",
        default=False,
        help="W_pred only used from c to z, not bidirectional (which causes weight transport)"
            "loss is a function of u = z*W_pred*drop_grad(c)",
    )
    group.add_option(
        "--freeze_W_pred",
        action="store_true",
        default=False,
        help="Boolean whether the k prediction weights W_pred (W_k in ContrastiveLoss) are frozen (require_grad=False).",
    )
    group.add_option(
        "--unfreeze_last_W_pred",
        action="store_true",
        default=False,
        help="Boolean whether the k prediction weights W_pred of the last module should be unfrozen.",
    )
    group.add_option(
        "--skip_upper_c_update",
        action="store_true",
        default=False,
        help="Boolean whether extra update in upper (context) layer is skipped. Consider this when predicting lower modules",
    )
    group.add_option(
        "--no_gamma",
        action="store_true",
        default=False,
        help="Boolean whether gamma (factor which sets the opposite sign of the update for pos and neg samples) is set to 1. i.e. third factor omitted in learning rule",
    )
    group.add_option(
        "--orthogonal_neg",
        action="store_true",
        default=False,
        help="force the loss to learn orthogonal representation rather than disaligned",
    )
    group.add_option(
        "--no_pred",
        action="store_true",
        default=False,
        help="Boolean whether Wpred * c is set to 1 (no prediction). i.e. fourth factor omitted in learning rule",
    )
    group.add_option(
        "--skip_step",
        type="int",
        default=1,
        help="number of skips to do for first prediction",
    )
    group.add_option(
        "--overlap_factor",
        type="int",
        default=2,
        help="division factors for downward moving patch",
    )
    group.add_option(
        "--res_stopgrad",
        action="store_true",
        default=False,
        help="Stop Gradient in ResBlock",
    )
    group.add_option(
        "--share_block",
        action="store_true",
        default=False,
        help="Share Subblocks within the split",
    )
    group.add_option(
        "--n_share",
        type="int",
        default=2,
        help="Number of blocks to share additionally",
    )
    group.add_option(
        "--share_only_final_block",
        action="store_true",
        default=False,
        help="Share Only the final block within the split",
    )
    group.add_option(
        "--time_embed",
        action="store_true",
        default=False,
        help="Add time embedding to shared block",
    )
    group.add_option(
        "--preact_block",
        action="store_true",
        default=False,
        help="use preactivation block",
    )
    group.add_option(
        "--gated_block",
        action="store_true",
        default=False,
        help="use gated block",
    )
    group.add_option(
        "--pool_block",
        action="store_true",
        default=False,
        help="use max pool to downsample block",
    )
    group.add_option(
        "--bottleneck",
        action="store_true",
        default=False,
        help="use bottleneck",
    )
    group.add_option(
        "--merge_conv1",
        action="store_true",
        default=False,
        help="whether merge conv1 in first loss",
    )
    group.add_option(
        "--conv1_dim",
        type="int",
        default=64,
        help="dimension of conv1",
    )
    group.add_option(
        "--conv1_kernel",
        type="int",
        default=5,
        help="dimension of conv1",
    )
    group.add_option(
        "--single_pool_block",
        action="store_true",
        default=False,
        help="use single convolution between residual, coupled with pooling",
    )
    group.add_option(
        "--pool_after_res",
        action="store_true",
        default=False,
        help="use single convolution between residual, coupled with pooling",
    )
    group.add_option(
        "--init_strategy",
        type="string",
        default='default',
        help="strategy from weight initialization, especially for shared block",
    )
    group.add_option(
        "--skip_downsample",
        action="store_true",
        default=False,
        help="do not use downsample at all, for first subblock, donnot add residual connection",
    )
    group.add_option(
        "--add_bias",
        action="store_true",
        default=False,
        help="add bias to conv",
    )
    group.add_option(
        "--rand_aug",
        action="store_true",
        default=False,
        help="use randaug crop for VGG like",
    )
    group.add_option(
        "--kernel_1",
        action="store_true",
        default=False,
        help="Make non dimensional modulating layer kernel 1",
    )
    group.add_option(
        "--loss_g",
        action="store_true",
        default=False,
        help="Additional loss on the g layer",
    )
    group.add_option(
        "--all_layer",
        action="store_true",
        default=False,
        help="Use all layers in a block",
    )
    group.add_option(
        "--pool_layer_loss",
        action="store_true",
        default=False,
        help="Mean pool the loss of layers in a block",
    )
    
    group.add_option(
        "--separate_g_pred",
        action="store_true",
        default=False,
        help="Use a separate W_pred for g",
    )
    
    group.add_option(
        "--adjusted_baseline",
        action="store_true",
        default=False,
        help="baseline adjusted for 1113 structure",
    )

    group.add_option(
        "--remove_final_maxpool",
        action="store_true",
        default=False,
        help="baseline adjusted for 1113 structure",
    )

    group.add_option(
        "--high_dim_exp",
        action="store_true",
        default=False,
        help="VGG high dimension (only 1024)",
    )

    group.add_option(
        "--greedy",
        action="store_true",
        default=False,
        help="do greedy training by evenly split the epochs to each module",
    )

    group.add_option(
        "--cum_greedy",
        action="store_true",
        default=False,
        help="do greedy training by evenly split the epochs to each module",
    )

    group.add_option(
        "--custom_greedy_ep",
        action="store_true",
        default=False,
        help="use custom epochs for greedy taining",
    )

    group.add_option(
        "--padding_mode",
        type="str",
        default="zeros",
        help="padding mode for CLAPP",
    )

    group.add_option(
        "--use_stride",
        action="store_true",
        default=False,
        help="use stride = 2 to replace max-pooling",
    )

    group.add_option(
        "--equilib_res",
        action="store_true",
        default=False,
        help="use equilibrium model type of residual: z^n = f(Wz^n-1 + z^0)",
    )

    group.add_option(
        "--hardsig_rec",
        action="store_true",
        default=False,
        help="use hard sigmoid min(1, max(x, 0))",
    )

    group.add_option(
        "--linear_inject",
        action="store_true",
        default=False,
        help="use linear injection for equilib: z^n = f(Wz^n-1 + Wx)",
    )

    group.add_option(
        "--unified_random_sampling",
        action="store_true",
        default=False,
        help="same random negative sampling for all layers",
    )

    group.add_option(
        "--use_transpose_pred",
        action="store_true",
        default=False,
        help="use the transpose for top-down prediction, only supporting one layer feedback!",
    )

    group.add_option(
        "--wm_amp", 
        type="float", 
        default=1e-6, 
        help="Learning rate WM amp"
    )

    group.add_option(
        "--wm_decay", 
        type="float", 
        default=1e-5, 
        help="Learning rate WM decay"
    )

    group.add_option(
        "--ia_null", 
        type="float", 
        default=0, 
        help="Learning rate IA null term"
    )
    group.add_option(
        "--phyll_theta", 
        type="float", 
        default=1.0, 
        help="Learning rate WM amp"
    )

    group.add_option(
        "--low_rank_dim", 
        type="int", 
        default=128, 
        help="Learning rate WM amp"
    )

    group.add_option(
        "--use_proj_head",
        action="store_true",
        default=False,
        help="project both z and c. But implemented as W_pred = W_1W_1",
    )
    group.add_option(
        "--use_asym_proj_head",
        action="store_true",
        default=False,
        help="project both z and c. But implemented as W_pred = W_1W_2, W1 != W2",
    )

    group.add_option(
        "--normalize_z",
        action="store_true",
        default=False,
        help="normalize the z and context, i.e. u-score is cos-similarity",
    )

    group.add_option(
        "--use_scheduler",
        action="store_true",
        default=False,
        help="Use Scheduler",
    )

    group.add_option(
        "--customize_loss_pool",
        type="str",
        default=None,
        help="str of customize pooling kernel for loss, could be 2-2-2-2-2-2"
    )

    group.add_option(
        "--adaptive_loss_pool",
        action="store_true",
        default=False,
        help="Use adaptive pooling for loss",
    )

    group.add_option(
        "--std_norm",
        action="store_true",
        default=False,
        help="Use standard normalization",
    )

    group.add_option(
        "--triangle_act",
        action="store_true",
        default=False,
        help="Use triangular activation",
    )

    group.add_option(
        "--batch_norm",
        action="store_true",
        default=False,
        help="Use batch normalization",
    )

    group.add_option(
        "--layer_norm",
        action="store_true",
        default=False,
        help="Use layer normalization",
    )

    group.add_option(
        "--normalize_pred",
        action="store_true",
        default=False,
        help="Normalize the prediction weights before each score computation",
    )

    group.add_option(
        "--update_proj_steps",
        type="int",
        default=0,
        help="(Number of) Time steps to predict into future",
    )

    group.add_option(
        "--retain_proj_grads",
        action="store_true",
        default=False,
        help="Retain the gradients of the projected features",
    )

    group.add_option(
        "--extra_lateral_loss",
        action="store_true",
        default=False,
        help="Use extra lateral loss; similar to pred_both, but with separate loss for lateral and top-down",
    )

    group.add_option(
        "--train_fb_with_grad",
        action="store_true",
        default=False,
        help="use gradients through the feedback weights to train feedback weights",
    )

    group.add_option(
        "--determined_neg_samples",
        action="store_true",
        default=True,
        help="use determined negative samples",
    )

    group.add_option(
        "--identity_projection",
        action="store_true",
        default=False,
        help="use identity projection for feedback weights",
    )


    return parser
