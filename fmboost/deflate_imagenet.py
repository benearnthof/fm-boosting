
# ImagenetInt8Dataset = StreamingDataset(
#     local=LOCAL_TRAIN_DIR,
#     remote=REMOTE_TRAIN_DIR,
#     split=None,
#     shuffle=True,
#     shuffle_algo="naive",
#     num_canonical_nodes=1,
#     batch_size = 32
# )



###### Example Usage. Decode back the 5th image. BTW shuffle plz


# batch = next(iter(train_dataloader))

# vae_latent = batch["vae_output"].reshape(-1, 4, 32, 32).cuda().float()

# # example decoding
# with torch.no_grad():
#     x = vae.decode(vae_latent.cuda()).sample
# # we don't have to save the images to disk
# # we have: high res inputs + hi res latents.
# # i think we stick to our dataset and just pass in hi res images.

# imgpath = Path("/workspace/datasets/imagenet")

# for idx in tqdm(range(0, 10, 1)):
#     # print(idx)
#     vae_latent = next(iter(train_dataloader))["vae_output"].reshape(-1, 4, 32, 32).cuda().float()
#     with torch.no_grad():
#         x = vae.decode(vae_latent.cuda()).sample
#     # img = VaeImageProcessor().postprocess(image = x.detach(), do_denormalize = [True, True])[0]
#     # img.save(imgpath / f"{idx:05d}.png")
from fmboost.dataloader import ImagenetDecodedDataset
import torch
REMOTE_TRAIN_DIR = "/workspace/datasets/vae_mds" # this is the path you installed this dataset.
LOCAL_TRAIN_DIR = "/workspace/datasets/local_train_dir"

from diffusers.models import AutoencoderKL
VAE = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda").eval()



ImagenetInt8Dataset = ImagenetDecodedDataset(
    local=LOCAL_TRAIN_DIR,
    remote=REMOTE_TRAIN_DIR,
    split="Train",
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size=32,
    vae=VAE
)

out = next(iter(ImagenetInt8Dataset))


train_dataloader = torch.utils.data.DataLoader(
    ImagenetInt8Dataset,
    batch_size=32,
    num_workers=0,
)

out = next(iter(train_dataloader))