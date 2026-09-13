"""Train VoidEnhancer from scratch on paired low/high-resolution images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from .losses import CharbonnierLoss, edge_loss
from .model import VoidEnhancer


class PairedImageDataset(Dataset):
    """Load matching LR/HR pairs and return aligned random patches."""

    def __init__(self, root: str, scale: int, patch_size: int = 128):
        self.root = Path(root)
        self.scale = scale
        self.patch_size = patch_size
        self.lr_dir = self.root / "lr"
        self.hr_dir = self.root / "hr"
        if not self.lr_dir.exists() or not self.hr_dir.exists():
            raise RuntimeError(f"Expected {self.lr_dir} and {self.hr_dir}")
        self.items = sorted(
            p for p in self.lr_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and (self.hr_dir / p.name).exists()
        )
        if not self.items:
            raise RuntimeError("No matching LR/HR image pairs found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        lr_path = self.items[index]
        hr_path = self.hr_dir / lr_path.name
        lr = TF.to_tensor(Image.open(lr_path).convert("RGB"))
        hr = TF.to_tensor(Image.open(hr_path).convert("RGB"))

        lr_h, lr_w = lr.shape[-2:]
        expected_h, expected_w = lr_h * self.scale, lr_w * self.scale
        if hr.shape[-2:] != (expected_h, expected_w):
            raise ValueError(f"Pair {lr_path.name} has LR {lr.shape[-2:]} and HR {hr.shape[-2:]}; expected {expected_h, expected_w}")

        ps = min(self.patch_size, lr_h, lr_w)
        if ps < 8:
            raise ValueError(f"Image {lr_path.name} is too small for training.")
        top = random.randint(0, lr_h - ps)
        left = random.randint(0, lr_w - ps)
        hr_top, hr_left = top * self.scale, left * self.scale
        hr_ps = ps * self.scale
        lr = lr[:, top:top + ps, left:left + ps]
        hr = hr[:, hr_top:hr_top + hr_ps, hr_left:hr_left + hr_ps]

        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoidEnhancer(scale=args.scale).to(device)
    loader = DataLoader(
        PairedImageDataset(args.data, args.scale, args.patch_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    reconstruction = CharbonnierLoss()
    model.train()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            pred = model(lr)
            loss = reconstruction(pred, hr) + args.edge_weight * edge_loss(pred, hr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        average = running / len(loader)
        print(f"epoch={epoch} loss={average:.6f} device={device}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "scale": args.scale, "epoch": epoch}, args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--output", default="checkpoints/voidenhancer.pt")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
