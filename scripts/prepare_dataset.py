"""Build paired LR/HR training images from a folder of high-quality images.

The model is trained from scratch, so this script creates the training pairs
without downloading a pretrained network. It simulates common low-quality video
artifacts: downscaling, blur, noise, JPEG compression and mild color changes.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def degrade(hr: Image.Image, scale: int) -> Image.Image:
    """Create a realistic low-quality counterpart for one HR image."""
    hr = hr.convert("RGB")
    width, height = hr.size
    lr_size = (max(8, width // scale), max(8, height // scale))

    lr = hr.resize(lr_size, Image.Resampling.LANCZOS)

    if random.random() < 0.65:
        lr = lr.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 1.1)))

    if random.random() < 0.45:
        lr = ImageEnhance.Contrast(lr).enhance(random.uniform(0.90, 1.08))
    if random.random() < 0.35:
        lr = ImageEnhance.Color(lr).enhance(random.uniform(0.92, 1.08))

    if random.random() < 0.75:
        quality = random.randint(35, 82)
        buffer = io.BytesIO()
        lr.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        lr = Image.open(buffer).convert("RGB")

    return lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing HR images")
    parser.add_argument("--output", default="data/train")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    source = Path(args.input)
    output = Path(args.output)
    hr_dir = output / "hr"
    lr_dir = output / "lr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
    random.shuffle(files)
    if args.limit:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No supported images found in {source}")

    for index, path in enumerate(files):
        try:
            with Image.open(path) as image:
                hr = image.convert("RGB")
                # Keep filenames unique and filesystem-safe.
                stem = f"{index:07d}"
                hr.save(hr_dir / f"{stem}.png", format="PNG")
                degrade(hr, args.scale).save(lr_dir / f"{stem}.png", format="PNG")
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    print(f"Prepared {len(list(hr_dir.glob('*.png')))} paired samples in {output}")


if __name__ == "__main__":
    main()
