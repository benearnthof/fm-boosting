apt update 
apt install -y tmux

apt install aria2 -y
apt update && apt install -y unzip
apt install -y parallel

apt-get update && apt-get install -y libgdcm-tools python3-gdcm

export HF_HOME=/workspace/checkpoints
export HF_HUB_ENABLE_HF_TRANSFER=True

cd /workspace/fm-boosting
bash gitconfig.sh

