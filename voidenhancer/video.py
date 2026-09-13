"""Video enhancement pipeline for trained VoidEnhancer checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .infer import load_model


def enhance_video(
    checkpoint: str,
    input_path: str,
    output_path: str,
    scale: int = 4,
    device_name: str = "auto",
) -> None:
    device = torch.device(
        "cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = load_model(checkpoint, scale, device)
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid dimensions.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_size = (width * scale, height * scale)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, target_size)
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create output video. Check OpenCV video codec support.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = pil_to_tensor(image).float().div(255.0).unsqueeze(0).to(device)
            with torch.inference_mode():
                enhanced = model(tensor).squeeze(0).cpu()
            out_rgb = cv2.cvtColor(__import__("numpy").array(to_pil_image(enhanced)), cv2.COLOR_RGB2BGR)
            writer.write(out_rgb)
            processed += 1
            if processed % 25 == 0:
                suffix = f"/{frame_count}" if frame_count else ""
                print(f"Processed {processed}{suffix} frames")
    finally:
        capture.release()
        writer.release()

    print(f"Saved enhanced video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    enhance_video(args.checkpoint, args.input, args.output, args.scale, args.device)


if __name__ == "__main__":
    main()
