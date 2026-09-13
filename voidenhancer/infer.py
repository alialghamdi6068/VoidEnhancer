"""Run the trained VoidEnhancer model on an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .model import VoidEnhancer


def load_model(checkpoint: str, scale: int, device: torch.device) -> VoidEnhancer:
    model = VoidEnhancer(scale=scale).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def enhance_image(model: VoidEnhancer, source: Path, destination: Path, device: torch.device) -> None:
    image = Image.open(source).convert("RGB")
    tensor = pil_to_tensor(image).float().div(255.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor).squeeze(0).cpu()
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(output).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = load_model(args.checkpoint, args.scale, device)
    enhance_image(model, Path(args.input), Path(args.output), device)
    print(f"Saved enhanced image to {args.output}")


if __name__ == "__main__":
    main()
