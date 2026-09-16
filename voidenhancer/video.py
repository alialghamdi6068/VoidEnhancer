"""Robust video enhancement pipeline for VoidEnhancer."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from .enhancement import EnhancementOptions, postprocess, preprocess
from .infer import load_model

Progress = Callable[[int, int], None]


def _mux_audio(video_only: Path, source: Path, output: Path) -> None:
    """Create a browser-friendly MP4 while retaining the source audio when possible."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        video_only.replace(output)
        return

    command = [
        ffmpeg, "-y",
        "-i", str(video_only),
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        # If the final encode fails, preserve the already-rendered video rather
        # than losing the completed job. Audio can be absent in this fallback.
        video_only.replace(output)


def enhance_video(checkpoint: str, input_path: str, output_path: str, scale: int = 4,
                  device_name: str = "auto", options: EnhancementOptions | None = None,
                  progress: Progress | None = None) -> None:
    options = options or EnhancementOptions(scale=scale)
    if scale not in (2, 4):
        raise ValueError("Scale must be 2 or 4.")
    source = Path(input_path)
    output = Path(output_path)
    device = torch.device("cuda" if device_name == "cuda" or
                          (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    model = load_model(checkpoint, scale, device)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid dimensions.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voidenhancer-") as temp:
        video_only = Path(temp) / "video.mp4"
        target_size = (width * scale, height * scale)
        writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), fps, target_size)
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create output video. Check codec support.")
        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = preprocess(frame, options)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = pil_to_tensor(Image.fromarray(rgb)).float().div(255.0).unsqueeze(0).to(device)
                with torch.inference_mode():
                    enhanced = model(tensor).squeeze(0).cpu().clamp(0, 1)
                out_rgb = np.asarray(to_pil_image(enhanced))
                out_bgr = postprocess(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR), options)
                if out_bgr.shape[1] != target_size[0] or out_bgr.shape[0] != target_size[1]:
                    out_bgr = cv2.resize(out_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
                writer.write(out_bgr)
                processed += 1
                if progress:
                    progress(processed, frame_count)
        finally:
            capture.release()
            writer.release()
        if processed == 0:
            raise RuntimeError("The video contains no readable frames.")
        _mux_audio(video_only, source, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    enhance_video(args.checkpoint, args.input, args.output, scale=args.scale, device_name=args.device)


if __name__ == "__main__":
    main()
