"""
model.py — DR Classification Model
=====================================
EfficientNetV2-S backbone with custom classification head
and built-in MC Dropout support for uncertainty estimation.

Architecture:
    - EfficientNetV2-S backbone
    - Global Average Pooling -> FC(1280,512) -> ReLU -> BN -> Dropout -> FC(512,5)

Usage:
    from src.model import DRClassifier, load_model
    model = DRClassifier()
    logits = model(images)
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional

from src.config import MODEL, PATHS


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy classifier based on EfficientNetV2-S.

    Architecture:
        - EfficientNetV2-S backbone (features only, num_classes=0)
        - Custom head: Dropout -> FC(1280,512) -> ReLU -> BN -> Dropout -> FC(512,5)
    """

    def __init__(
        self,
        num_classes: int = None,
        dropout_rate: float = None,
        pretrained: bool = None,
    ):
        super().__init__()

        num_classes = num_classes or MODEL.num_classes
        self.dropout_rate = dropout_rate or MODEL.dropout_rate
        pretrained = pretrained if pretrained is not None else MODEL.pretrained

        self.num_classes = num_classes

        # Load EfficientNetV2-S backbone (features only, no classifier)
        self.backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=pretrained, num_classes=0)

        self.feature_dim = self.backbone.num_features  # 1280 for EfficientNetV2-S

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),       # head.0
            nn.Linear(self.feature_dim, 512),      # head.1
            nn.ReLU(inplace=True),                 # head.2
            nn.BatchNorm1d(512),                   # head.3
            nn.Dropout(p=self.dropout_rate),       # head.4
            nn.Linear(512, num_classes),           # head.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns logits.
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def enable_mc_dropout(self):
        """Enable dropout layers while keeping BatchNorm in eval mode."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def get_target_layer(self):
        """Return the target layer for Grad-CAM++."""
        # For timm tf_efficientnetv2_s, the final convolution before pooling is conv_head
        if hasattr(self.backbone, 'conv_head'):
            return self.backbone.conv_head
        # Fallback for other architectures
        return None


def build_model(device: torch.device = None) -> DRClassifier:
    """Build a new DRClassifier and move to device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DRClassifier()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Model: EfficientNetV2-S + Custom Head")
    print(f"Total parameters: {total_params:,}")
    print(f"  Backbone (EfficientNetV2-S): {backbone_params:,}")
    print(f"  Head: {head_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")

    return model


def load_model(
    checkpoint_path: Optional[str] = None,
    device: torch.device = None,
) -> DRClassifier:
    """
    Load the trained DRClassifier from the active deployment checkpoint.
    Strictly loads retinaai_efficientnetv2s_deployment.pth with strict=True.
    No fallback allowed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = str(PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Mandatory active model checkpoint not found at {ckpt_path}. "
            "Fallbacks to old models or uninitialized weights are disabled."
        )

    model = DRClassifier(pretrained=False)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Enforce strict state dict loading — no fallback
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded active DRClassifier model weights successfully from {ckpt_path.name} (strict=True).")

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    print("Model module loaded successfully.")
    model = build_model(torch.device("cpu"))
    model.eval()  # Set eval mode so BatchNorm1d supports single-sample (batch size 1) forward pass
    x = torch.randn(1, 3, 256, 256)
    logits = model(x)
    print(f"Input: {x.shape} -> Output: {logits.shape}")
