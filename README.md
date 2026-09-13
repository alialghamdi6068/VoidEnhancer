# VoidEnhancer

VoidEnhancer is a from-scratch AI video enhancement project.

## Goal

Train our own neural network to reconstruct higher-quality video from low-resolution/compressed input (for example 460p), with a future target of 1080p/4K output.

## Important

This project does **not** use a pretrained upscaling model as its model architecture. The neural network in `voidenhancer/model.py` is implemented from scratch using PyTorch. PyTorch is the training framework, not a pretrained enhancement model.

## Structure

```text
VoidEnhancer/
├── voidenhancer/
│   ├── __init__.py
│   ├── model.py          # Our neural network
│   ├── dataset.py        # Video frame dataset
│   ├── losses.py         # Training losses
│   └── train.py          # Training entry point
├── scripts/
│   └── prepare_dataset.py
├── web/
│   └── README.md         # Website architecture will be added after the model pipeline works
├── requirements.txt
└── README.md
```

## First milestone

Train a 2x/4x image super-resolution model on paired low/high quality frames. Once the core model is stable, extend it to temporal video processing and 4K inference.
