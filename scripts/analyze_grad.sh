stl10_root='YOUR_DATASET_FOLDER/stl10'

for i in 99
do
    echo "Extracting and analyzing gradient from epoch $i"

     python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_stl10 --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
    python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_dfb_stl10 --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_fb_idx 6-6-6-6-6-6 --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
    python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_dfb_stl10_train_fb_mse --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_fb_idx 6-6-6-6-6-6 --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
done