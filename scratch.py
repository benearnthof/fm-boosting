from diffusers import StableDiffusionPipeline
import torch

from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL

diffusers_ae = DiffusersAutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae")

torch.save(diffusers_ae, "/root/checkpoints/sd_ae.ckpt")

from omegaconf import OmegaConf
# train.py passes config argument as string
cfg = OmegaConf.load("./fm-boosting/configs/flow400_64-128/unet-base_psu.yaml")
# command line arguments that are missing from cfg:
cfg.data

from fmboost.helpers import get_obj_from_str, instantiate_from_config
loader = get_obj_from_str("fmboost.dataloader.DataModuleFromConfig")
loader = instantiate_from_config(cfg.data)

# loader is setup with train, validation, test dicts
loader.setup()
# loader.datasets is a dict of datasets that get instantiated in a loop
# so loader.train, loader.validation, loader.test are three different configs
# these configs get passed into loader.setup

dl = loader._train_dataloader()

x = next(iter(dl))
x.shape

cfg.model.params.first_stage_cfg
fmboost_ae = instantiate_from_config(cfg.model.params.first_stage_cfg)
fmboost_ae = fmboost_ae.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae").cuda()

from diffusers.models import AutoencoderKL as DiffusersAutoencoderKL
isinstance(fmboost_ae, DiffusersAutoencoderKL)

# quick encode decode check
x_latent = fmboost_ae.encode(x.to("cuda"))
# hack for posterior of original VAE
x_latent = x_latent.latent_dist.sample()
x_latent.shape

x_recon = fmboost_ae.decode(x_latent).sample
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