# local-SSL
**This is the official reporistory for the paper:** [Call Local Learning Match Self-Supervised Backpropagation?](https://www.arxiv.org/abs/2601.21683)

The repository provides code to train convolutional neural networks with local self-supervised learning rules that are biologically plausible and completely backpropagation free. 

# Installation

## Conda Installation
**Remark:** We provide the python and package versions used to produce numbers in the paper. More recent versions also work and produce similar simulation results.
1. Create conda environment with python 3.9:
```
conda create -n local-ssl python=3.9
conda activate local-ssl
```
2. Install pytorch 2.0.1 and torchvision 0.15.2 ([official website](https://pytorch.org/get-started/previous-versions/)). 
```
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
3. Install other dependencies
```
pip install wandb tqdm pyyaml numpy==1.26.4 matplotlib==3.9.4
```

## DockerFile

Use the ``Dockerfile`` provided in the repo:
```
docker build . --platform linux/amd64  --tag my-local-ssl --build-arg LDAP_USERNAME=YOUR_USER_NAME
```

Essentially, we just use PyTorch images from Nvidia. For EPFL users, a block of code (commented out in the Dockerfile) is provided to run on the EPFL computing clusters.

# Tutorials
Two Tuturials are provided for you to understand the code

1. The notebook ``train_local_ssl.ipynb`` demonstrates how to train a simple CNN model using the local-SSL objective.

2. The notebook ``linear_nn.ipynb`` uses code for Figure 2 as an example to explain how to use the theory codes to study local-SSL in linear networks.

# Reproduce Experiments
## Image Benchmarks:
Exact codes are provided in the ``./scripts`` folder. In general you could train the model using the following line (CLAPP++DFB on STL10 as an example):
```
python -m vision.train_ssl --save_dir clapp_dfb_stl10 --data_input_dir YOUR_STL10_DATASET_PATH --random_crop_size 96 --num_epochs 300 --asymmetric_W_pred --unified_random_sampling --customize_loss_pool 12-12-12-6-6-3 --customize_fb_idx 6-6-6-6-6-6
```

The simulation results are stored in ``./logs`` by default, but you could customize through the argument ``--data_output_dir``. Then, you could evaluate the model by using the following code to train a downstream linear classifier:
```
python -m vision.eval_ssl --model_path ./logs/clapp_stl10 --data_input_dir $stl10_root --random_crop_size 96 --model_num 299 --num_epochs 300 --in_channels 32768 --multi_module_num 1-2-3-4-5-6 --asymmetric_W_pred --customize_loss_pool 12-12-12-6-6-3 --no_eval_patch_average --validate
```

## Meaning of the important arguments:
The meaning are explained based on the notations in the paper. For minor arguments, refer to files in the ``./arg_parser`` folder.
1. ``--contrast mode``: the function f
2. ``--asymmetric_W_pred``: W_pred here means the projection matrix $B^l$. Asymentric means that context $c^l$ has been detached from auto-differentiation graph and gradient only flow 'asymmetrically' through the term $(z^l)^T B^l c^l$. You should always set this to true to ensure the updates are local.
3. ``--customize_loss_pool``: the size of local spatial averaging pooling in each layer. If not specified, the spatial dimensions are globally averaged.
4. ``--customize_fb_idx``: the layer where the context $c^l$ comes from. For DFB, all are from the last layer, so it is '6-6-6-6-6-6'

## Theory simulations:
The code for Figure 2, 3a, 3b, and 4a are all put in the ``theory`` folder. You should ``cd`` into the ``theory`` folder and then find simulation codes in ``simulations.sh`` file. By default, the simulation results will be saved in the folder ``theory/stats``, and corresponding models will be in ``theory/theory_models``. 

To make the plots, just run the following line in the ``theory`` folder, where ``FIGURE_NUM`` could be 2, 3a, 3b, or 4a. Make sure you specify the correct path to simulation results (by default: ``theory/stats``) in ``plot_figures.py``
```
python plot_figures.py FIGURE_NUM
```
