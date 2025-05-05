import tarfile
import io
import os
import subprocess
import tempfile
import random
from webdataset import ShardWriter
from pathlib import Path
import pandas as pd
import json
import re

# Config
SOURCE_TAR = "ldm100k-data/LDM-data/LDM_100k.tar"
assert Path(SOURCE_TAR).exists(), f"Source tar file {SOURCE_TAR} not found"
SAMPLES_PER_SHARD = 32
GCS_BUCKET = "gs://ldm100k-bucket/shards/"
SHARD_TEMPLATE = "data-%06d.tar"
SHUFFLE_BUFFER_SIZE = 512  # In-memory shuffle buffer size
METADATA_PATH = Path("/home/Bene/ldm100k-data/metadata/LDM_100k/participants.tsv")

def upload_shard(fname):
    os.system(f"gsutil cp {fname} {GCS_BUCKET}")
    os.unlink(fname)


def main():
    meta = pd.read_csv(METADATA_PATH, sep="\t", index_col="participant_id")
    # Stream from the tar
    with tarfile.open(SOURCE_TAR, "r") as tar:
        # members = tar.getmembers()[0:20]
        # member = members[-1]
        # fileobj = tar.extractfile(member)
        # data = fileobj.read()
        # output_path = "/home/Bene/ldm100k-data/metadata/LDM_100k/testfile.nii.gz"
        # with open(output_path, "wb") as f:
        #     f.write(data)
        buffer = []
        shard_id = 0
        sample_id = 0

        for member in tar:
            if not member.isfile():
                continue

            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            data = fileobj.read()

            # Check if the file is a .nii.gz file
            if member.name.endswith(".nii.gz"):
                # Build sample with the .nii.gz binary data
                # Extract subject ID, e.g., sub-001234
                match = re.search(r"sub-\d{6}", member.name)
                if not match:
                    continue  # Skip files without valid subject ID

                subject_id = match.group(0)

                if subject_id not in meta.index:
                    continue  # Skip if no metadata

                # Convert metadata row to dict
                metadata_dict = meta.loc[subject_id].to_dict()

                sample = {
                    "__key__": f"{sample_id:08d}",
                    "image.nii.gz": data,
                    "json": json.dumps(metadata_dict)  # stores as .json in the shard
                }
                buffer.append(sample)
                sample_id += 1
                if shard_id+1 % 10 == 0:
                    print(f"Current Shard: {shard_id} @ Shardsize of {SAMPLES_PER_SHARD}")

            # Shuffle and flush if buffer is full
            if len(buffer) >= SHUFFLE_BUFFER_SIZE:
                random.shuffle(buffer)
                while len(buffer) >= SAMPLES_PER_SHARD:
                    shard_samples = [buffer.pop() for _ in range(SAMPLES_PER_SHARD)]
                    write_and_upload_shard(shard_samples, shard_id)
                    shard_id += 1

        # Final flush
        if buffer:
            random.shuffle(buffer)
            while len(buffer) >= SAMPLES_PER_SHARD:
                shard_samples = [buffer.pop() for _ in range(SAMPLES_PER_SHARD)]
                write_and_upload_shard(shard_samples, shard_id)
                shard_id += 1

            if buffer:
                # leftover final shard
                write_and_upload_shard(buffer, shard_id)

def write_and_upload_shard(samples, shard_id):
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, f"shard-{shard_id}%06d.tar")
        # print(shard_path)
        with ShardWriter(pattern=shard_path, maxcount=len(samples)) as sink:
            for sample in samples:
                sink.write(sample)

        print(f"Uploading shard {shard_id}...")
        # get the actual path the writer saved the data to disk in
        print(len(os.listdir(tmpdir)))
        tarfile = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        # can only be one file since we unlink the file after every upload
        upload_shard(tarfile)


if __name__ == "__main__":
    main()


# OUTPUT_DIR = Path("ldm100k-data/metadata")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Files to extract
# METADATA_FILES = {
#     "LDM_100k/dataset_description.json",
#     "LDM_100k/participants.json",
#     "LDM_100k/participants.tsv",
# }

# with tarfile.open(SOURCE_TAR, "r") as tar:
#     for member in tar.getmembers():
#         if member.name in METADATA_FILES:
#             print(f"Extracting {member.name}")
#             tar.extract(member, path=OUTPUT_DIR)
