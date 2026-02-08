# local-SSL
Reporistory for Paper: Call Local Learning Match Self-Supervised Backpropagation?

# Installation

## Conda Installation
1. Create conda environment with python 3.9:
```
conda create -n local-ssl python=3.9
conda activate local-ssl
```
2. Install pytorch 2.0.1 and torchvision 0.15.2 ([official website](https://pytorch.org/get-started/previous-versions/)). Latest versions should also be able to run the code, but exact numbers may differ from reported ones in the paper.
```
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
3. Install other dependencies
```
pip install numpy==1.26.4 matplotlib==3.9.4
```
# Variables:
1. function f as 'contrast mode'
2. W_pred is B^l

