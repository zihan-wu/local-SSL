timgnet_root='YOUR_TINY_IMAGENET_DATASET_PATH' #


# BP-CLAPP++ training and downstream evaluation, NO spatial dependence in last layer
python -m vision.train_ssl --save_dir clapp_ete_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 4 --ete_training --weight_decay 0.00001
python -m vision.eval_ssl --model_path ./logs/clapp_ete_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate

# CLAPP++ training and downstream evaluation, NO spatial dependence
python -m vision.train_ssl --save_dir clapp_pooled_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --num_epochs 300 --asymmetric_W_pred --weight_decay 0.00001
python -m vision.eval_ssl --model_path ./logs/clapp_pooled_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate

# CLAPP++ training and downstream evaluation, with spatial dependence
python -m vision.train_ssl --save_dir clapp_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --weight_decay 0.00001
python -m vision.eval_ssl --model_path ./logs/clapp_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate

# CLAPP++DFB training and downstream evaluation, with spatial dependence and direct feedback
python -m vision.train_ssl --save_dir clapp_dfb_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --customize_fb_idx 6-6-6-6-6-6 --weight_decay 0.00001
python -m vision.eval_ssl --model_path ./logs/clapp_dfb_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate

# CLAPP++both training and downstream evaluation, with spatial dependence from same layer and direct feedback
python -m vision.train_ssl --save_dir clapp_both_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --customize_fb_idx 6-6-6-6-6-6 --extra_lateral_loss --weight_decay 0.00001
python -m vision.eval_ssl --model_path ./logs/clapp_both_timagenet --dataset tiny_imagenet --data_input_dir $timgnet_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate


