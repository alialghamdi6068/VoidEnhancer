"""Train VoidEnhancer from scratch on paired low/high-resolution images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from .losses import CharbonnierLoss, edge_loss
from .model import VoidEnhancer


class PairedImageDataset(Dataset):
    """Loads matching LR/HR images from data/train/lr and data/train/hr."""

    def __init__(self, root: str, patch_size: int = 192, scale: int = 4):
        self.root = Path(root)
        self.lr_dir = self.root / "lr"
        self.hr_dir = self.root / "hr"
        self.patch_size = patch_size
        self.scale = scale
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

        expected_hr = (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
        if tuple(hr.shape[-2:]) != expected_hr:
            raise ValueError(f"Pair {lr_path.name} has incompatible LR/HR dimensions.")

        max_lr_h = min(lr.shape[-2], self.patch_size)
        max_lr_w = min(lr.shape[-1], self.patch_size)
        top = torch.randint(0, lr.shape[-2] - max_lr_h + 1, (1,)).item()
        left = torch.randint(0, lr.shape[-1] - max_lr_w + 1, (1,)).item()
        lr = lr[:, top:top + max_lr_h, left:left + max_lr_w]
        hr = hr[:, top * self.scale:(top + max_lr_h) * self.scale,
                left * self.scale:(left + max_lr_w) * self.scale]
        return lr, hr


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoidEnhancer(scale=args.scale).to(device)
    loader = DataLoader(
        PairedImageDataset(args.data, args.patch_size, args.scale),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pixel_loss = CharbonnierLoss()
    best_loss = float("inf")
    model.train()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            loss = pixel_loss(pred, hr) + args.edge_weight * edge_loss(pred, hr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        average = running / len(loader)
        print(f"epoch={epoch} loss={average:.6f} device={device}")
        if average < best_loss or epoch % args.save_every == 0:
            best_loss = min(best_loss, average)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "scale": args.scale,
                    "epoch": epoch,
                    "loss": average,
                },
                args.output,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--output", default="checkpoints/voidenhancer.pt")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
