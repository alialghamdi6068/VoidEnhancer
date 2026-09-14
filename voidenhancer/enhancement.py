"""Adaptive, dependency-light enhancement stages used by VoidEnhancer.

The trained VoidEnhancer model remains the primary learned stage. These stages
are deterministic and deliberately conservative so a bad input does not turn
into an over-sharpened output.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementOptions:
    denoise: bool = True
    deblock: bool = True
    sharpen: bool = True
    color: bool = True
    scale: int = 4


def _unsharp(frame: np.ndarray, amount: float = 0.28) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def _color_recovery(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray, options: EnhancementOptions) -> np.ndarray:
    """Conservative cleanup before the learned super-resolution stage."""
    result = frame
    if options.denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 2.0, 2.0, 7, 21)
    if options.deblock:
        # A light bilateral pass reduces ringing/block edges without destroying detail.
        result = cv2.bilateralFilter(result, d=5, sigmaColor=18, sigmaSpace=18)
    if options.color:
        result = _color_recovery(result)
    return result


def postprocess(frame: np.ndarray, options: EnhancementOptions) -> np.ndarray:
    result = frame
    if options.sharpen:
        result = _unsharp(result, 0.18 if options.scale == 4 else 0.14)
    return np.clip(result, 0, 255).astype(np.uint8)


def write_manifest(job_dir: Path, options: EnhancementOptions) -> None:
    (job_dir / "pipeline.txt").write_text(
        "VoidEnhancer adaptive pipeline\n"
        f"scale={options.scale}\n"
        f"denoise={options.denoise}\n"
        f"deblock={options.deblock}\n"
        f"sharpen={options.sharpen}\n"
        f"color={options.color}\n",
        encoding="utf-8",
    )
