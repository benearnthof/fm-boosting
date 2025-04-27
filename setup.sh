cd /root
cd /workspace

git clone https://github.com/benearnthof/fm-boosting.git
# dataset

mkdir ./datasets
cd ./datasets
mkdir ./local_train_dir # for imagenet
mkdir ./imagenet
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

sudo apt-get install pigz
tar -I pigz -xvzf 102flowers.tgz --no-same-owner

tar -xvzf 102flowers.tgz --no-same-owner

# for imagenet.int8
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=True
pip install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./vae_mds
pip install mosaicml-streaming

cd /root
cd /workspace
cd fm-boosting
bash gitconfig.sh

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
pip uninstall numpy -y
pip install numpy==1.26.4
# pip install -U tensorboardX
pip install -U tensorboard
# pip install xformers

# saving pretrained klautoencoder
cd ..
mkdir ./checkpoints 
# HF home
# if we're in workspace this functions like /root for our commands
export HF_HOME=/workspace/checkpoints

# TODO: Create secrets.yaml in fm-boosting and paste wandbapikey: