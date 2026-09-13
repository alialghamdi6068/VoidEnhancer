"""Evaluate a trained VoidEnhancer checkpoint on paired LR/HR images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from voidenhancer.model import VoidEnhancer


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lr", required=True)
    parser.add_argument("--hr", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    scale = int(checkpoint["scale"])
    model = VoidEnhancer(scale=scale)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lr = TF.to_tensor(Image.open(args.lr).convert("RGB")).unsqueeze(0)
    hr = TF.to_tensor(Image.open(args.hr).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        pred = model(lr)

    if pred.shape != hr.shape:
        raise SystemExit(f"Shape mismatch: prediction={tuple(pred.shape)} target={tuple(hr.shape)}")

    print(f"PSNR: {psnr(pred, hr):.3f} dB")
    print(f"L1:   {torch.mean(torch.abs(pred - hr)).item():.6f}")


if __name__ == "__main__":
    main()
