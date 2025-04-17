cd /root

git clone https://github.com/benearnthof/fm-boosting.git
# dataset
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz

pip install omegaconf
pip install webdataset
pip install pytorch-lightning

pip install diffusers["torch"] transformers
# saving pretrained klautoencoder
mkdir -p /root/checkpoints

# HF home
export HF_HOME=/root/checkpoints

