import os
import shutil
import random
random.seed(42)
# Source directory (where all folders are now)
source_dir = "/workspace/datasets/qure.headct.study"

# Destination directory for validation set
val_dir = "/workspace/datasets/qure.headct.val"
os.makedirs(val_dir, exist_ok=True)

# Get all subdirectories, filtering only patient folders (e.g., CQ500-CT-0, etc.)
all_cases = [d for d in os.listdir(source_dir) 
             if d.startswith("CQ500-CT-") and os.path.isdir(os.path.join(source_dir, d))]

# Randomly select 90 for validation
val_cases = random.sample(all_cases, 90)

# Move selected folders to validation directory
for case in val_cases:
    src_path = os.path.join(source_dir, case)
    dst_path = os.path.join(val_dir, case)
    shutil.move(src_path, dst_path)  # Use shutil.copytree if you want to copy instead
    print(f"Moved {case} to validation set")

print("✅ Done: Moved 90 cases to validation set.")
