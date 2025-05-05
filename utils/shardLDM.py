import tarfile
import io
import os
import subprocess
import tempfile
import random
from webdataset import ShardWriter
from pathlib import Path

# Config
SOURCE_TAR = "ldm100k-data/LDM-data/LDM_100k.tar"
Path(SOURCE_TAR).exists()
SAMPLES_PER_SHARD = 32
GCS_BUCKET = "gs://ldm100k-bucket/shards"
SHARD_TEMPLATE = "data-%06d.tar"
SHUFFLE_BUFFER_SIZE = 512  # In-memory shuffle buffer size

def upload_to_gcs(local_path):
    subprocess.run(["gsutil", "-q", "cp", local_path, GCS_BUCKET], check=True)
    os.remove(local_path)

def main():
    # Stream from the tar
    with tarfile.open(SOURCE_TAR, "r") as tar:
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

            # Build sample
            ext = member.name.split(".")[-1]
            sample = {
                "__key__": f"{sample_id:08d}",
                ext: data
            }
            buffer.append(sample)
            sample_id += 1

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
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmpf:
        tmpf.close()
        with ShardWriter(tmpf.name, maxcount=len(samples)) as sink:
            for sample in samples:
                sink.write(sample)

        print(f"Uploading shard {shard_id}...")
        upload_to_gcs(tmpf.name)

if __name__ == "__main__":
    main()
