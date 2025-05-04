cd /root
cd /workspace

# git clone https://github.com/benearnthof/fm-boosting.git
# # dataset

# mkdir ./datasets
# cd ./datasets
# mkdir ./local_train_dir # for imagenet
# mkdir ./imagenet
# # wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# tar -xvzf 102flowers.tgz --no-same-owner
# rm 102flowers.tgz

# venv on persistent storage
# cd /workspace
# python3 -m venv venv
source /workspace/venv/bin/activate
# for imagenet.int8
# pip install hf_transfer
# pip install -U "huggingface_hub[cli]"
# huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./vae_mds
# pip install mosaicml-streaming
# pip install omegaconf
# pip install webdataset
# pip install pytorch-lightning
# pip install diffusers["torch"] transformers
# pip install wandb
# pip install einops
# pip install torchdiffeq
# pip install torchmetrics
# pip install torch-fidelity
# pip install albumentations
# pip install matplotlib
# pip install open_clip_torch
# pip install lpips
# pip install pytorch-fid
# pip install -U tensorboardX
# pip install -U tensorboard
# pip install xformers
# https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256
# pip install datasets

# guarantee np version
# pip uninstall numpy -y
# pip install numpy==1.26.4

# saving pretrained klautoencoder
# cd ..
# mkdir ./checkpoints 
# HF home
# if we're in workspace this functions like /root for our commands
export HF_HOME=/workspace/checkpoints
export HF_HUB_ENABLE_HF_TRANSFER=True

apt update 
apt install -y tmux

apt install aria2 -y
# QURE headstudy CT
# aria2c "magnet:?xt=urn:btih:47e9d8aab761e75fd0a81982fa62bddf3a173831&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"


# downsampled open images
#magnet:?xt=urn:btih:9208d33aceb2ca3eb2beb70a192600c9c41efba1&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
aria2c --meta-download <magnet-link>
aria2c --show-files <magnet-link>
aria2c --select-file=2 -d /path/to/download/folder <magnet-link>



# bs 32, 16 workers spu: 17, 1.25it/s

cd /workspace/fm-boosting
bash gitconfig.sh
# TODO: Verify raw setup