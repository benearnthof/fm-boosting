# goal: Specify a dataset for dicom files that
# either returns individual slices
# or returns stacks of three consecutive slices extracted from volumes 
# they are then passed to the flowmatching model so encoding with KLautoencoder 
# can be done optionally.

# three options: 
# concatenate 3 single channel slices
# zeropad single channel to 3 channel inuts => ruled out as probably too noisy
# replace encoder.encoder.conv_in with single channel conv2d
#   could average model weights then finetune for a bit
# what about using 3 slices and converting to greyscale in the end? 
#   input 3 channels => encode => decode => greyscale => denormalization
#   this is not good for upscaling cause of spatial influences maybe? maybe it is beneficial?
# what about stacking a single slice three times? 
#   we could stack => encode => decode => greyscale => denormalize

import os
import glob
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from functools import partial

QURE_ROOT = Path(r"D:\Torrents\QureHeadCT")

root_dir = QURE_ROOT
image_size = 256
series = []
exts = ['.dcm']

# first approach: Pseudo 3 channel images by stacking one image 3 times
class QureHeadCTDataset(Dataset):
    """
    Dataset that takes in a folder of uncompressed .dcm images and returns
    {"image": tensor, "latent", tensor}
    """
    def __init__(
        self,
        root_dir: str | Path,
        image_size: int,
        exts: List[str] = ['.dcm'],
        augment_horizontal_flip = False,
        convert_image_to = None,
        window_size=3
    ):
        super().__init__()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        assert root_dir.is_dir()

        self.root_dir = root_dir
        self.image_size = image_size

        self.maybe_convert_fn = partial(self.convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(self.maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        # Preprocess: find all bottom-most directories containing .dcm files
        subdirs = [x for x in self.root_dir.glob('*/**/') if any(x.glob('*.dcm'))]
        self.dicom_paths = [subdir/f for f in os.listdir(subdir) if f.endswith('.dcm') for subdir in subdirs]
    
    def convert_image_to_fn(self, img_type, image):
        if image.mode == img_type:
            return image
        return image.convert(img_type)

    def __len__(self):
        return len(self.dicom_paths))

    def __getitem__(self, index):
        path = self.dicom_paths[index]
        dicom_data = pydicom.dcmread(path)
        pixel_array = dicom_data.pixel_array.astype(np.float32)
        # clip and normalize
        hu_min, hu_max = -1000, 1000
        pixel_array = np.clip(pixel_array, hu_min, hu_max)
        pixel_array = ((pixel_array / 1000))
        # Image.fromarray(((pixel_array + 1) * 127.5).clip(0,255).astype(np.uint8)).convert('L').save('output2.png')
        image = Image.fromarray(pixel_array)
        image = self.transform(image)
        stacked_image = image.repeat(3, 1, 1)
        # T.ToPILImage()(stacked_image).save('stacked_fauxcolor.png')
        # stacked_np = (stacked_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)  # Shape: (H, W, 3)
        # Image.fromarray(stacked_np).convert('L').save('stacked_grayscale.png')
        return {"image": stacked_image}


import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL


encoder = AutoencoderKL.from_pretrained("sd-legacy/stable-diffusion-v1-5", subfolder="vae")

encoder.eval()

with torch.no_grad():
    latent = encoder.encode(stacked_image.unsqueeze(0))

latent = latent.latent_dist.sample()
latent.shape

with torch.no_grad():
    recon = encoder.decode(latent).sample

recon.shape


# Unnormalize from [-1, 1] to [0, 255]
recon_img = ((recon.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
Image.fromarray(recon_img).save('reconstructed_faux_color.png')

Image.fromarray(recon_img[:, :, 1]).convert('L').save('reconstructed_middle_channel_grayscale.png')
