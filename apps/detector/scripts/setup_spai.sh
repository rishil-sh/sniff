#!/usr/bin/env bash
# One-time install of SPAI (CVPR 2025, Apache-2.0) into its own venv.
# Its pinned deps (timm 0.4.12, numpy 1.26) clash with ours, so it runs as a subprocess.
set -euo pipefail

DIR="${AI_DETECTOR_CACHE:-$(cd "$(dirname "$0")/.." && pwd)/data/models}/spai"
COMMIT=8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4
WEIGHTS_ID=1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI

if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/mever-team/spai.git "$DIR"
fi
git -C "$DIR" fetch --depth 1 origin "$COMMIT"
git -C "$DIR" checkout -q "$COMMIT"

uv venv --allow-existing --python 3.11 "$DIR/.venv"
# Inference subset of requirements.txt; training-only extras (neptune, tensorboard, apex, CLIP) are skipped.
uv pip install --python "$DIR/.venv/bin/python" \
  torch==2.3.1 torchvision==0.18.1 "numpy~=1.26.4" "timm==0.4.12" "yacs~=0.1.8" "einops~=0.8.0" \
  "albumentations==1.4.14" "albucore==0.0.16" "opencv-python~=4.10.0.84" "pillow~=10.4.0" \
  "scipy~=1.14.0" "pandas~=2.2.2" "seaborn~=0.13.2" filetype lmdb networkx click tqdm pyyaml \
  "termcolor~=2.4.0" "torchmetrics~=1.4.0" neptune tensorboard onnx onnxscript

mkdir -p "$DIR/weights"
if [ ! -s "$DIR/weights/spai.pth" ]; then
  curl -L --fail -o "$DIR/weights/spai.pth" \
    "https://drive.usercontent.google.com/download?id=${WEIGHTS_ID}&export=download&confirm=t"
fi
echo "SPAI ready at $DIR"
