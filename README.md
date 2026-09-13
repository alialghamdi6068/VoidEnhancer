# VoidEnhancer

VoidEnhancer is a from-scratch AI video enhancement project.

## Goal

Train our own neural network to reconstruct higher-quality video from low-resolution and compressed input, then run that model on complete videos. A 4x model can produce 460p → roughly 1840p output; the final quality depends on what the model learned from the training data. Upscaling cannot recover information that was never present exactly.

## AI approach

- PyTorch is the framework; there is no pretrained upscaler inside the model.
- `voidenhancer/model.py` contains the neural network implemented from scratch.
- `voidenhancer/losses.py` contains reconstruction and edge losses.
- Training uses paired high-quality images and synthetic low-quality versions.
- `voidenhancer/video.py` applies the trained model frame by frame to video.
- Temporal consistency is a later model milestone; it is not faked by calling the single-frame model a video model.

## Project structure

```text
VoidEnhancer/
├── voidenhancer/
│   ├── __init__.py
│   ├── model.py          # From-scratch neural network
│   ├── losses.py         # Reconstruction + edge losses
│   ├── train.py          # Training with random paired patches
│   ├── infer.py          # Single-image inference
│   └── video.py          # Video frame inference
├── scripts/
│   └── prepare_dataset.py # Build LR/HR pairs from HR images
├── web/
│   ├── app.py            # FastAPI website/API
│   ├── templates/index.html
│   └── static/style.css
├── requirements.txt
└── README.md
```

## Training workflow

1. Put authorized high-quality training images in a local folder, for example `dataset/hr_source/`.
2. Build synthetic LR/HR pairs:

```bash
python scripts/prepare_dataset.py --input dataset/hr_source --output data/train --scale 4
```

3. Train the model:

```bash
python -m voidenhancer.train --data data/train --scale 4 --epochs 50 --batch-size 4 --patch-size 128
```

4. The checkpoint is written to `checkpoints/voidenhancer.pt`.

## Image inference

```bash
python -m voidenhancer.infer --checkpoint checkpoints/voidenhancer.pt --input input.png --output enhanced.png --scale 4
```

## Video inference

```bash
python -m voidenhancer.video --checkpoint checkpoints/voidenhancer.pt --input input.mp4 --output output.mp4 --scale 4
```

## Website

Start the API after a trained checkpoint exists:

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

The website uses the same inference code as the CLI. It does not contain a separate fake or placeholder AI pipeline.

## Current status

The code pipeline is in place, but **there are no trained production weights in the repository yet**. Training a useful model requires a real dataset and compute time. The next model milestone after the baseline is temporal consistency for video, followed by validation on real compressed clips.
