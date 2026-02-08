source /lcncluster/zihan/.caas_HOME/miniconda3/bin/activate
conda init
conda activate local-ssl
cd /lcncluster/zihan/local-SSL

python -m vision.train_ssl --save_dir clapp_dfb_stl10 --random_crop_size 96 --num_epochs 300 --asymmetric_W_pred --unified_random_sampling --customize_loss_pool 12-12-12-6-6-3 --customize_fb_idx 6-6-6-6-6-6
python -m vision.eval_ssl --model_path ./logs/clapp_dfb_stl10 --random_crop_size 96 --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate