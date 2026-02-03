cifar_root='YOUR_CIFAR10_DATASET_PATH'
timgnet_root='YOUR_TINY_IMAGENET_DATASET_PATH'
stl10_root='YOUR_STL10_DATASET_PATH'

# BP-SSL, SimCLR style InfoNCE

# STL10 Training
python -m vision.train_ssl --random_crop_size 96 --data_input_dir $stl10_root --save_dir simclr_ete_stl10 --contrast_mode 'infonce' --num_epochs 300 --customize_loss_pool 6 --ete_training --low_rank_dim 1024
python -m vision.eval_ssl --random_crop_size 96 --data_input_dir $stl10_root --model_path ./logs/simclr_ete_stl10 --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate --low_rank_dim 1024

# Tiny ImageNet Training
python -m vision.train_ssl --dataset tiny_imagenet --data_input_dir $timgnet_root --save_dir simclr_ete_timagenet --contrast_mode 'infonce' --num_epochs 300 --customize_loss_pool 4 --ete_training --low_rank_dim 1024
python -m vision.eval_ssl --dataset tiny_imagenet --data_input_dir $timgnet_root --model_path ./logs/simclr_ete_timagenet --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --customize_loss_pool 8-8-8-4-4-2 --no_eval_patch_average --validate --low_rank_dim 1024

# CIFAR10 Training
python -m vision.train_ssl --dataset cifar10 --data_input_dir $cifar_root --save_dir simclr_ete_cifar --contrast_mode 'infonce' --num_epochs 300 --customize_loss_pool 2 --ete_training --low_rank_dim 1024
python -m vision.eval_ssl --dataset cifar10 --data_input_dir $cifar_root --model_path ./logs/simclr_ete_cifar --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --customize_loss_pool 4-4-4-2-2-1 --no_eval_patch_average --validate --low_rank_dim 1024
