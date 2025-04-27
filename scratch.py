from diffusers import StableDiffusionPipeline
import torch

from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
import os
from pathlib import Path
os.getenv("HF_HOME")
Path(os.getenv("HF_HOME")).exists()

# diffusers_ae = DiffusersAutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae")
# torch.save(diffusers_ae, "/workspace/checkpoints/sd_ae.ckpt")

from omegaconf import OmegaConf
# train.py passes config argument as string
cfg = OmegaConf.load("/workspace/fm-boosting/configs/flow400_64-128/unet-base_psu.yaml")
cfg = OmegaConf.load("/workspace/fm-boosting/configs/flow400_64-128/unet-base_psu_imagenet.yaml")
cfg = OmegaConf.load("/workspace/fm-boosting/configs/flow400_64-128/unet-base_psu_imagenet256.yaml")
# command line arguments that are missing from cfg:
cfg.data

from fmboost.helpers import get_obj_from_str, instantiate_from_config
# loader = get_obj_from_str("fmboost.dataloader.DataModuleFromConfig")

xd = instantiate_from_config(cfg.data)
xd.setup()


# loader is setup with train, validation, test dicts
# loader.datasets is a dict of datasets that get instantiated in a loop
# so loader.train, loader.validation, loader.test are three different configs
# these configs get passed into loader.setup

# cfg.model.params.first_stage_cfg
# fmboost_ae = instantiate_from_config(cfg.model.params.first_stage_cfg)
# fmboost_ae = fmboost_ae.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae").cuda()

from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
# isinstance(fmboost_ae, DiffusersAutoencoderKL)


# Now how do we get this to train? 
# Investigate train.py

# where is this used?
# from fmboost.helpers import load_model_weights

# train args:
# class train_args:
#     def __init__(self):
#         self.config = "/workspace/fm-boosting/configs/flow400_64-128/unet-base_psu.yaml"
#         self.name = "runpod_test"
#         self.resume_checkpoint = None
#         self.load_weights = None
#         self.num_nodes = 1
#         self.devices = -1
#         self.find_unused_parameters = "ddp_notebook"
#         self.p2p_disable = False
#         self.seed = 2025
#         self.tqdm_refresh_rate = 1
#         self.use_wandb = True
#         self.use_wandb_offline = False

# args = train_args()

# cfg = OmegaConf.load(args.config)

module = instantiate_from_config(cfg.model)


diffusers_ae = DiffusersAutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae").cuda()

module.first_stage.to("cuda")
# loader = instantiate_from_config(cfg.data)
# loader.setup()
dl = xd._train_dataloader()

batch = next(iter(dl))
batch["image"].shape
x = batch["image"][0].unsqueeze(0)
# quick encode decode check
x_latent = module.first_stage.encode(x.to("cuda"))
# hack for posterior of original VAE
x_latent = x_latent.latent_dist.sample()
x_latent.shape

x_recon =  module.first_stage.decode(x_latent).sample
x_recon.shape

import torchvision.transforms as T
from PIL import Image
import torch

orig_pil = T.ToPILImage()(x.squeeze(0).cpu())

# Decode tensor to image
decoded_clamped = torch.clamp(x_recon.squeeze(0).cpu(), 0, 1)
decoded_pil = T.ToPILImage()(decoded_clamped)

# Save both
orig_pil.save("original_image.png")
decoded_pil.save("decoded_image.png")


import torch
def are_state_dicts_equal(model1, model2):
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    if sd1.keys() != sd2.keys():
        return False

    for key in sd1:
        if not torch.equal(sd1[key], sd2[key]):
            return False
    return True

are_state_dicts_equal(module.first_stage, diffusers_ae)



# TODO: precompute embeddings
# 5-10 epochs on oxford flowers already yields really good results.
# Larger image dataset
# TODO: aria2c for academic torrent of CT images
# TODO: preprocess CT image dataset into nice slices/custom dataloader.
# CT images/Single channel images
# Check if the latents of the autoencoder are meaningful 
# TODO: check sampling from checkpoint
# TODO: use tmux to train so ssh session doesnt crash
# TODO: triplecheck training parameters from paper
# TODO: integrate with other models 
# TODO: build docker container
# TODO: Eval on test set



# import torchvision.transforms as T
# from fmboost.dataloader import HuggingfaceImagenet256

# wrapped_dataset = HuggingfaceImagenet256(
#     split="train",
#     image_size=256,
#     augment_horizontal_flip=True,
#     convert_image_to=None
# )

# sample = wrapped_dataset[482749]
# print(sample["image"].shape)  # should be torch.Size([3, 256, 256])

# T.ToPILImage()(sample["image"]).save("sample.jpg")