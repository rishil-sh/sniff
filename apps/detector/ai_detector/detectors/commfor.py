"""Community Forensics ViT-S (Park & Owens, CVPR 2025). MIT.

Architecture matches JeongsooP/Community-Forensics `ViTClassifier`.
Weights: OwensLab/commfor-model-384. Instantiated on CPU so Mac/MPS can load.
"""

from __future__ import annotations

import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    def __init__(self, model_size: str = "small", input_size: int = 384, patch_size: int = 16):
        super().__init__()
        names = {
            ("small", 224, 16): "vit_small_patch16_224.augreg_in21k_ft_in1k",
            ("small", 224, 32): "vit_small_patch32_224.augreg_in21k_ft_in1k",
            ("small", 384, 16): "vit_small_patch16_384.augreg_in21k_ft_in1k",
            ("small", 384, 32): "vit_small_patch32_384.augreg_in21k_ft_in1k",
            ("tiny", 224, 16): "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            ("tiny", 384, 16): "vit_tiny_patch16_384.augreg_in21k_ft_in1k",
        }
        name = names[(model_size, input_size, patch_size)]
        self.vit = timm.create_model(name, pretrained=False)
        head_in = 192 if model_size == "tiny" else 384
        self.vit.head = nn.Linear(head_in, 1)

    def forward(self, x):
        return self.vit(x)
