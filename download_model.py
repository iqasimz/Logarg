#!/usr/bin/env python
from huggingface_hub import snapshot_download
import shutil
import os

REPO_ID    = "iqasimz/logarg-relationtagger"
TARGET_DIR = "models/relationtagger"

def main():
    # remove any existing files so we get a clean download
    if os.path.isdir(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    # snapshot_download will pull down all files (including LFS) into cache_dir or local_dir
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False
    )

    print(f"✔ Model snapshot downloaded to '{TARGET_DIR}'")

if __name__ == "__main__":
    main()