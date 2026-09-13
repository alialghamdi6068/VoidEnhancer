"""Download DIV2K training/validation archives for VoidEnhancer.

This script only downloads the public dataset archive. It does not contain
model weights and does not use a pretrained enhancement model.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URLS = {
    "train_hr": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_hr": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(archive: Path, destination: Path) -> None:
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/div2k")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    root = Path(args.output)
    archives = root / "archives"
    extracted = root / "extracted"

    train = archives / "DIV2K_train_HR.zip"
    valid = archives / "DIV2K_valid_HR.zip"
    if not train.exists():
        download(URLS["train_hr"], train)
    if not valid.exists():
        download(URLS["valid_hr"], valid)

    train_target = extracted / "train"
    valid_target = extracted / "valid"
    if not train_target.exists():
        extract_zip(train, train_target)
    if not valid_target.exists():
        extract_zip(valid, valid_target)

    if not args.keep_archives:
        shutil.rmtree(archives)

    print(f"Dataset ready under {extracted}")


if __name__ == "__main__":
    main()
