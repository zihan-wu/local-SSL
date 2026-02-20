### Simulations for Figure 2 in the paper ###
# With all assumptions / Random Fixed B 
python run_fastb.py --model_name 128x6_linear_lw_fastb_orthogonal
# Non-orthogonal W 
python run_fastb.py --model_name 128x6_linear_lw_fastb_default --config_override custom_init=default
# ReLU MLP
python run_fastb.py --model_name 128x6_relu_lw_fastb_orthogonal --config_override linear_nn=False
# ReLU MLP with non-orthogonal W
python run_fastb.py --model_name 128x6_relu_lw_fastb_default --config_override linear_nn=False custom_init=default


### Simulations for Figure 3a in the paper ###
# Linear f without feedback: (the word 'linear' in model_name means linearNN, not linear f)
python run_fastb.py --model_name 128to4_linear_lw_fastb_orthogonal_l2 --config_override contrast_mode=linear layer_dim=[128,64,32,16,8,4]
# Linear f with feedback (DFB):
python run_fastb.py --model_name 128to4_linear_fb_fastb_orthogonal_l2 --config_override contrast_mode=linear layer_dim=[128,64,32,16,8,4] fb_idx=[5,5,5,5,5,5]
# Softplus f without feedback:
python run_fastb.py --model_name 128to4_linear_lw_fastb_orthogonal_logsigl2 --config_override layer_dim=[128,64,32,16,8,4]
# Softplus f with feedback (DFB):
python run_fastb.py --model_name 128to4_linear_fb_fastb_orthogonal_logsigl2 --config_override layer_dim=[128,64,32,16,8,4] fb_idx=[5,5,5,5,5,5]


### Simulations for Figure 3b in the paper ###
# local-SSL without feedback:
python run_mnist.py --model_name clapp_lw_512x6_logsigl2_seed42 --config_override fb_idx=[0,1,2,3,4,5]
# local-SSL DFB:
python run_mnist.py --model_name clapp_fb_512x6_logsigl2_seed42
# local-SSL, theoretical optimal update:
python run_mnist.py --model_name clapp_fb_512x6_train_fb_with_grad_logsigl2_seed42 --config_override train_fb_with_grad=True
# local-SSL, random fixed feedback:
python run_mnist.py --model_name clapp_fb_512x6_frozen_fb_logsigl2_seed42 --config_override freeze_fb=True 


### Simulations for Figure 4b in the paper ###
# Linear CNN without spatial dependence:
python run_fastb.py --model_config ./configuration_linear_cnn.yaml --model_name cnnlinear_32x4_lwp_fastb_orthogonal_logsigl2 --config_override custom_pool=[8,4,2,1] 
# Linear CNN with spatial dependence:
python run_fastb.py --model_config ./configuration_linear_cnn.yaml --model_name cnnlinear_32x4_lw_fastb_orthogonal_logsigl2
# Linear CNN with spatial dependence + DFB:
python run_fastb.py --model_config ./configuration_linear_cnn.yaml --model_name cnnlinear_32x4_dfb_fastb_orthogonal_logsigl2 --config_override fb_idx=[3,3,3,3]


