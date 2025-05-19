
# mkdir ./local_train_dir # for imagenet
# mkdir ./imagenet
# # wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

# tar -xvzf 102flowers.tgz --no-same-owner
# rm 102flowers.tgz

# QURE headstudy CT
cd /workspace/datasets
aria2c "magnet:?xt=urn:btih:47e9d8aab761e75fd0a81982fa62bddf3a173831&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
cd ./qure.headct.study

# this takes a couple minutes
parallel -j 8 'unzip -q {} -d {=s/.zip//=} && rm {}' ::: *.zip

#ls *.zip | parallel -j 8 'unzip -q {} -d {=s/.zip//=} && rm {}'
# for f in *.zip; do unzip -q "$f" -d "${f%.zip}" && rm "$f"; done

# venv on persistent storage

# for imagenet.int8
export MAKEFLAGS="-j$(nproc)"
pip install --upgrade pip setuptools wheel

pip install hf_transfer
pip install -U "huggingface_hub[cli]"
# huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./vae_mds

pip install mosaicml-streaming
# pip install -j $(nproc) mosaicml-streaming

pip install omegaconf
pip install webdataset
pip install --install-option="--jobs=6" pytorch-lightning
pip install pytorch-lightning
pip install diffusers["torch"] transformers
pip install einops
pip install torchdiffeq
pip install torchmetrics
pip install torch-fidelity
pip install albumentations
pip install matplotlib
pip install open_clip_torch
pip install lpips
pip install pytorch-fid
pip install -U tensorboardX
pip install -U tensorboard
pip install xformers
# https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256
pip install datasets

# required for pydicom on VM

pip install --upgrade pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
pip install pydicom

# guarantee np version
# pip uninstall wandb -y
# pip install wandb==0.19.8

# pip uninstall numpy;
# rm -rI numpy;

pip uninstall numpy -y
pip install numpy==1.26.4
pip list | grep numpy
# saving pretrained klautoencoder
# cd ..
# mkdir ./checkpoints 
# HF home
# if we're in workspace this functions like /root for our commands

# downsampled open images
#magnet:?xt=urn:btih:9208d33aceb2ca3eb2beb70a192600c9c41efba1&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
# aria2c --meta-download <magnet-link>
# aria2c --show-files <magnet-link>
# aria2c --select-file=2 -d /path/to/download/folder <magnet-link>

# LDM 100k
# aria2c --bt-metadata-only=true --bt-save-metadata=true "magnet:?xt=urn:btih:63aeb864bbe2115ded0aa0d7d36334c026f0660b&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php%3Fpasskey%3D59191383faf97bc1bf5459852ce2acef&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce"
# downloading via .torrent is faster since BEP-9 is very slow 
# aria2c --bt-metadata-only=true \
#        --bt-save-metadata=true \
#        --show-files=true \
#        /workspace/datasets/LDM-data-63aeb864bbe2115ded0aa0d7d36334c026f0660b.torrent


# bs 32, 16 workers spu: 17, 1.25it/s


# TODO: Verify raw setup

# EMPTY PIP CACHE
