# code to load images contiguously and save them as slices in all 3 directions to disk
import os
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import zoom
import psutil

def get_safe_worker_count():
    physical_cores = psutil.cpu_count(logical=False)
    return max(1, physical_cores - 1)  # Leave 1 core free

# process one time for train set
# one time for val set
QURE_ROOT = Path(r"/workspace/datasets/qure.headct.val")
root_dir = QURE_ROOT
image_size = 256

subdirs = [x for x in root_dir.glob('*/**/') if any(x.glob('*.dcm'))]
# dicom_paths = [f for subdir in subdirs for f in Path(subdir).glob('*.dcm')]


def load_dicom_series(folder):
    dicom_files = sorted(Path(folder).glob('*.dcm'), key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    volume = np.stack([s.pixel_array for s in slices])
    return volume


def pad_volume_to_depth(volume, target_depth=256):
    depth = volume.shape[0]
    if depth >= target_depth:
        return volume
    pad_total = target_depth - depth
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=-2000)
    return padded

# padded = pad_volume_to_depth(normalized)

def center_crop_2d(slice_, target_size=256):
    h, w = slice_.shape
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    return slice_[start_h:start_h + target_size, start_w:start_w + target_size]

import pydicom
from pydicom.uid import generate_uid
from copy import deepcopy
from pydicom.uid import ExplicitVRLittleEndian


def save_augmented_slices_as_dicom(volume, original_dicom, output_dir, volume_index):
    output_dir = Path(output_dir)
    volume = pad_volume_to_depth(volume, target_depth=256)
    orig_slice_count = volume.shape[0]
    axes_to_augment = [1, 2]  # Coronal, Sagittal

    for axis in axes_to_augment:
        slices = np.moveaxis(volume, axis, 0)
        axis_dir = output_dir / str(volume_index) / str(axis)
        axis_dir.mkdir(parents=True, exist_ok=True)

        # Only keep `orig_slice_count` centered slices, downsampled
        start = (slices.shape[0] - orig_slice_count) // 2
        end = start + orig_slice_count
        selected_slices = slices[start:end][::2]  # every second slice

        for i, slice_array in enumerate(selected_slices):
            cropped = center_crop_2d(slice_array, 256).astype(np.int16)
            # Copy original metadata and update
            new_dcm = deepcopy(original_dicom)
            new_dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            new_dcm.is_implicit_VR = False
            new_dcm.is_little_endian = True
            new_dcm.PixelData = cropped.tobytes()
            new_dcm.Rows, new_dcm.Columns = cropped.shape
            new_dcm.InstanceNumber = i + 1
            new_dcm.SOPInstanceUID = generate_uid()
            new_dcm.file_meta.MediaStorageSOPInstanceUID = new_dcm.SOPInstanceUID
            new_dcm.save_as(axis_dir / f"{i:04d}.dcm", write_like_original=False)


from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback

def process_volume(folder_idx_pair):
    idx, folder = folder_idx_pair
    try:
        dicom_files = sorted(Path(folder).glob('*.dcm'), key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
        if len(dicom_files) < 128:
            return None  # Skip
        
        slices = [pydicom.dcmread(f) for f in dicom_files]
        volume = np.stack([s.pixel_array for s in slices])
        reference_dcm = slices[len(slices) // 2]  # Use middle slice for metadata
        save_augmented_slices_as_dicom(volume, reference_dcm, Path(r"/workspace/datasets/qure.headct.val/augmented"), idx)
        return idx  # Success
    except Exception as e:
        print(f"[{idx}] Error processing {folder}:\n{e}\n{traceback.format_exc()}")
        return None

def process_all_parallel(subdirs, max_workers=6):
    folder_idx_pairs = list(enumerate(subdirs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_volume, pair) for pair in folder_idx_pairs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing volumes"):
            pass

if __name__ == "__main__":
    root_dir = QURE_ROOT
    subdirs = [x for x in root_dir.glob('*/**/') if any(x.glob('*.dcm'))]
    print(len(subdirs))
    process_all_parallel(subdirs)
