# code to load images contiguously and save them as slices in all 3 directions to disk
import os
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import zoom

QURE_ROOT = Path(r"D:\Torrents\QureHeadCT")
root_dir = QURE_ROOT
image_size = 256
series = []
exts = ['.dcm']

subdirs = [x for x in root_dir.glob('*/**/') if any(x.glob('*.dcm'))]
# dicom_paths = [f for subdir in subdirs for f in Path(subdir).glob('*.dcm')]

n_slices = [len(os.listdir(x)) for x in subdirs]

subdirs[0:10]

n_slices[0:10]


def load_dicom_series(folder):
    dicom_files = sorted(Path(folder).glob('*.dcm'), key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    volume = np.stack([s.pixel_array for s in slices])
    return volume

def normalize_volume(volume):
    hu_min, hu_max = -1000, 1000
    volume = np.clip(volume, hu_min, hu_max)
    volume = volume / 1000.0  # Now in range ~[-1, 1]
    return volume

# folder = subdirs[0]
# volume = load_dicom_series(folder)
# normalized = normalize_volume(volume)
# normalized.shape

def pad_volume_to_depth(volume, target_depth=256):
    depth = volume.shape[0]
    if depth >= target_depth:
        return volume
    pad_total = target_depth - depth
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=-1.0)
    return padded

# padded = pad_volume_to_depth(normalized)

def center_crop_2d(slice_, target_size=256):
    h, w = slice_.shape
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return slice_[start_h:start_h + target_size, start_w:start_w + target_size]

# slice_ = padded[:, 128, :]
# cropped = center_crop_2d(slice_)

# debug_png = ((cropped+ 1.0) / 2.0 * 255.0).astype(np.uint8)
# Image.fromarray(debug_png).save("cropped.png")

# output_dir = Path("D:\Torrents\QureHeadAugmented")

def save_augmented_slices(volume, output_dir, volume_index):
    output_dir = Path(output_dir)
    volume = pad_volume_to_depth(volume, target_depth=256)
    orig_slice_count = volume.shape[0]
    axes_to_augment = [1, 2]  # Coronal, Sagittal

    for axis in axes_to_augment:
        slices = np.moveaxis(volume, axis, 0)
        axis_dir = output_dir / str(volume_index) / str(axis)
        axis_dir.mkdir(parents=True, exist_ok=True)

        # Only keep `orig_slice_count` centered slices
        start = (slices.shape[0] - orig_slice_count) // 4
        end = start + orig_slice_count
        slices = slices[start:end][::2] # downsampling to avoid oversampling augmented images

        for i, slice_ in enumerate(slices):
            cropped = center_crop_2d(slice_, 256)
            np.save(axis_dir / f"{i:04d}.npy", cropped)


from tqdm import tqdm

def process_all(subdirs, output_dir=Path(r"D:\Torrents\QureHeadAugmented")):
    for idx, folder in enumerate(tqdm(subdirs, desc="Processing volumes")):
        try:
            num_slices = len([f for f in os.listdir(folder) if f.endswith('.dcm')])
            if num_slices < 128:
                continue
            volume = load_dicom_series(folder)
            volume = normalize_volume(volume)
            save_augmented_slices(volume, output_dir, idx)
        except Exception as e:
            print(f"[{idx}] Error processing {folder}: {e}")



process_all(subdirs)
