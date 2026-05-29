imgnet_root='YOUR_IMAGENET_DATASET_PATH' 


NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
# DDP is only used for SSL training, not used for downstream evaluation. For evaluation, we used only one GPU

# BP-CLAPP++ training and downstream evaluation, NO spatial dependence in last layer
torchrun --nproc_per_node=$NPROC -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_ete_imagenet_vgg8 --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 100 --log_pos_neg --distr_strategy ddp --ete_training --use_scheduler --learning_rate 0.3--weight_decay 0.000001
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 128 --model_path ./logs/clapp_pooled_imagenet_vgg8 --model_num 99 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 100 --in_channels 5504 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --validate

# CLAPP++ training and downstream evaluation, NO spatial dependence
torchrun --nproc_per_node=$NPROC  -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_pooled_imagenet_vgg8 --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 100 --log_pos_neg --distr_strategy ddp --use_scheduler --weight_decay 0.000001 --learning_rate 0.3 
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 128 --model_path ./logs/clapp_pooled_imagenet_vgg8 --model_num 99 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 100 --in_channels 5504 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --validate

# CLAPP++ training and downstream evaluation, with spatial dependence
torchrun --nproc_per_node=$NPROC  -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_imagenet_vgg8 --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 100 --log_pos_neg --distr_strategy ddp --use_scheduler --weight_decay 0.000001 --learning_rate 0.3 --customize_loss_pool 8-4-4-4-2-2-1-1 --adaptive_loss_pool
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 128 --model_path ./logs/clapp_imagenet_vgg8 --model_num 99 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 100 --in_channels 5504 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --validate

# CLAPP++ training and downstream evaluation, with spatial dependence and direct feedback
torchrun --nproc_per_node=$NPROC  -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_dfb_imagenet_vgg8 --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 100 --log_pos_neg --distr_strategy ddp --use_scheduler --weight_decay 0.000001 --learning_rate 0.3 --customize_fb_idx 8-8-8-8-8-8-8-8 --customize_loss_pool 8-4-4-4-2-2-1-1 --adaptive_loss_pool
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 128 --model_path ./logs/clapp_dfb_imagenet_vgg8 --model_num 99 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 100 --in_channels 5504 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --validate --customize_fb_idx 8-8-8-8-8-8-8-8

# CLAPP++ training and downstream evaluation, with spatial dependence from same layer and direct feedback
torchrun --nproc_per_node=$NPROC  -m vision.train_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 64 --save_dir clapp_both_imagenet_vgg8 --model_splits 8 --train_module 8 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 100 --log_pos_neg --distr_strategy ddp --use_scheduler --weight_decay 0.000001 --learning_rate 0.3 --customize_fb_idx 8-8-8-8-8-8-8-8 --customize_loss_pool 8-4-4-4-2-2-1-1 --adaptive_loss_pool --extra_lateral_loss
python -m vision.eval_ssl --dataset imagenet --data_input_dir $imgnet_root --batch_size 128 --model_path ./logs/clapp_both_imagenet_vgg8 --model_num 99 --model_splits 8 --train_module 8 --module_num 8 --num_epochs 100 --in_channels 5504 --multi_module_num 1-2-3-4-5-6-7-8 --asymmetric_W_pred --validate --customize_fb_idx 8-8-8-8-8-8-8-8 --extra_lateral_loss