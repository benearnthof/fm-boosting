#!/bin/bash

# Parse the API key from secrets.yaml using yq (preferred) or grep/sed fallback
KEY=$(yq '.wandbapikey' /workspace/fm-boosting/secrets.yaml 2>/dev/null)

# Fallback method using grep/sed if yq isn't available
if [ -z "$KEY" ]; then
    KEY=$(grep 'wandbapikey:' /workspace/fm-boosting/secrets.yaml | sed 's/wandbapikey:[ ]*//')
fi

# Log in to wandb
wandb login "$KEY"

# Now start training
python train.py --config=/workspace/fm-boosting/configs/flow400_64-128/unet-base_psu.yaml --use_wandb
