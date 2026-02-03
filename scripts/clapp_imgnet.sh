imgnet_root='YOUR_IMAGENET_DATASET_PATH' 


export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
# we obtain approximately 35% for ImageNet
python -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_dfb_imagenet_lars_sche --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 50 --distr_strategy ddp --use_scheduler --customize_fb_idx 8-8-8-8-8-8-8-8 --weight_decay 0.0001 --learning_rate 0.6 --customize_loss_pool 28-28-14-14-14-7-7-7 
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 32 --model_path ./logs/clapp_dfb_imagenet_lars_sche --model_num 49 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 50 --in_channels 35840 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --customize_loss_pool 28-28-28-14-14-7-7-7 --no_eval_patch_average --validate


