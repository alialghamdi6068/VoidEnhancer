"""Train VoidEnhancer from scratch on paired low/high-resolution images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from .model import VoidEnhancer


class PairedImageDataset(Dataset):
    """Loads matching LR/HR images from data/train/lr and data/train/hr."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.lr_dir = self.root / "lr"
        self.hr_dir = self.root / "hr"
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
        return lr, hr


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoidEnhancer(scale=args.scale).to(device)
    loader = DataLoader(
        PairedImageDataset(args.data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pixel_loss = nn.L1Loss()
    model.train()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            if pred.shape[-2:] != hr.shape[-2:]:
                raise ValueError(
                    f"HR size {tuple(hr.shape[-2:])} does not match model output "
                    f"{tuple(pred.shape[-2:])}. Check the scale and dataset pairs."
                )
            loss = pixel_loss(pred, hr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        average = running / len(loader)
        print(f"epoch={epoch} loss={average:.6f} device={device}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "scale": args.scale,
                    "epoch": epoch,
                },
                args.output,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--output", default="checkpoints/voidenhancer.pt")
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
