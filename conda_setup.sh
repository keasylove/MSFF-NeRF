#!/bin/bash
eval "$(conda shell.bash hook)"
conda create -n msff-nerf python=3.8 -y

conda activate msff-nerf

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py38_cu113_pyt1121.tar.bz2
pip install -r requirements.txt
