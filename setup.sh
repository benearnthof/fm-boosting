cd /root
cd /workspace

git clone https://github.com/benearnthof/fm-boosting.git
# dataset

mkdir ./datasets
cd ./datasets
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz --no-same-owner

# TOOD: replace with requirements.txt
pip install omegaconf
pip install webdataset
pip install pytorch-lightning
pip install diffusers["torch"] transformers
pip install wandb
pip install einops
pip install torchdiffeq
pip install torchmetrics
pip install torch-fidelity
pip install albumentations
pip install matplotlib
pip install open_clip_torch
pip install lpips
pip install pytorch-fid
# pip install xformers

# saving pretrained klautoencoder
cd ..
mkdir ./checkpoints 
# HF home
# if we're in workspace this functions like /root for our commands
export HF_HOME=/workspace/checkpoints

