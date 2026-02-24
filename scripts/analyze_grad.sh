stl10_root='YOUR_DATASET_FOLDER/stl10'

# Model for gradient alignment check: last layer no spatial dependence
python -m vision.train_ssl --save_dir clapp_stl10_poollast --data_input_dir $stl10_root --random_crop_size 96 --num_epochs 300 --asymmetric_W_pred --unified_random_sampling --customize_loss_pool 12-12-12-6-6-6
python -m vision.train_ssl --save_dir clapp_dfb_stl10_poollast --data_input_dir $stl10_root --random_crop_size 96 --num_epochs 300 --asymmetric_W_pred --unified_random_sampling --customize_loss_pool 12-12-12-6-6-6 --customize_fb_idx 6-6-6-6-6-6
# Train explicitly to match BP gradients, determined negative samples are used for stable training
python -m vision.train_ssl --save_dir clapp_dfb_stl10_train_fb_mse --data_input_dir $stl10_root --random_crop_size 96 --num_epochs 50 --asymmetric_W_pred --unified_random_sampling --customize_loss_pool 12-12-12-6-6-6 --customize_fb_idx 6-6-6-6-6-6 --train_fb_with_grad --batch_size 128


for i in 99
do
    echo "Extracting and analyzing gradient from epoch $i"

     python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_stl10 --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
    python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_dfb_stl10 --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_fb_idx 6-6-6-6-6-6 --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
    python -m vision.analyze_gradient --data_input_dir $stl10_root --save_dir gradient_analysis --model_path ./logs/clapp_dfb_stl10_train_fb_mse --model_num $i --random_crop_size 96 --asymmetric_W_pred --customize_fb_idx 6-6-6-6-6-6 --customize_loss_pool 12-12-12-6-6-6 --reload_index_path ./logs/rand_index_ssl/ --unified_random_sampling
done