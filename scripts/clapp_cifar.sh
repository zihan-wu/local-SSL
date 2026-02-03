
cifar_root='YOUR_STL10_DATASET_PATH'

# BP-CLAPP++ training and downstream evaluation, NO spatial dependence in last layer
python -m vision.train_ssl --save_dir clapp_ete_cifar --dataset cifar10 --data_input_dir $cifar_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 2 --ete_training
python -m vision.eval_ssl --model_path ./logs/clapp_ete_cifar --dataset cifar10 --data_input_dir $cifar_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 4-4-4-2-2-1 --no_eval_patch_average --validate

# CLAPP++ training and downstream evaluation, NO spatial dependence
python -m vision.train_ssl --save_dir clapp_pooled_cifar --dataset cifar10 --data_input_dir $cifar_root --num_epochs 300 --asymmetric_W_pred
python -m vision.eval_ssl --model_path ./logs/clapp_pooled_cifar --dataset cifar10 --data_input_dir $cifar_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 4-4-4-2-2-1 --no_eval_patch_average --validate

# CLAPP++ training and downstream evaluation, with spatial dependence
python -m vision.train_ssl --save_dir clapp_cifar --dataset cifar10 --data_input_dir $cifar_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3
python -m vision.eval_ssl --model_path ./logs/clapp_cifar --dataset cifar10 --data_input_dir $cifar_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate

# CLAPP++DFB training and downstream evaluation, with spatial dependence and direct feedback
python -m vision.train_ssl --save_dir clapp_dfb_cifar --dataset cifar10 --data_input_dir $cifar_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --customize_fb_idx 6-6-6-6-6-6
python -m vision.eval_ssl --model_path ./logs/clapp_dfb_cifar --dataset cifar10 --data_input_dir $cifar_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate

# CLAPP++both training and downstream evaluation, with spatial dependence from same layer and direct feedback
python -m vision.train_ssl --save_dir clapp_both_cifar --dataset cifar10 --data_input_dir $cifar_root --num_epochs 300 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --customize_fb_idx 6-6-6-6-6-6 --extra_lateral_loss
python -m vision.eval_ssl --model_path ./logs/clapp_both_cifar --dataset cifar10 --data_input_dir $cifar_root --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate


