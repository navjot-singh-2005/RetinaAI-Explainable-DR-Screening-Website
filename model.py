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


import os
import shutil
import urllib.request
from typing import Optional, Union

DEPLOYMENT_CHECKPOINT_NAME = "retinaai_efficientnetv2s_deployment.pth"
EXPECTED_CHECKPOINT_SHA256 = "ef4eecbe959884403ac86a91c3386f992d5c23ebb3b2076b0797aab8269541c5"


def ensure_checkpoint_available(
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Ensure the production checkpoint is available locally on disk.
    If missing (e.g. fresh clone on Streamlit Community Cloud where GitHub's
    25MB browser limit prevented committing the 84MB weights), automatically
    downloads it once from Hugging Face Hub or a configured MODEL_URL and caches it.

    Secrets / Environment variables supported:
      - HF_MODEL_REPO: Hugging Face repo ID (e.g. "username/retinaai-model")
      - HF_MODEL_FILENAME: Filename on HF repo (default: "retinaai_efficientnetv2s_deployment.pth")
      - HF_TOKEN: Optional auth token for private HF repositories
      - MODEL_URL: Direct HTTPS download URL (GitHub Releases, HF resolve URL, S3, etc.)
    """
    if checkpoint_path is None:
        checkpoint_path = PATHS.model_final / DEPLOYMENT_CHECKPOINT_NAME

    ckpt_path = Path(checkpoint_path).resolve()

    # 1. If checkpoint already exists locally and is > 10MB, return cached file immediately
    if ckpt_path.exists() and ckpt_path.stat().st_size > 10_000_000:
        return ckpt_path

    # Ensure parent directory exists
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Check for environment variables or Streamlit secrets
    hf_repo = os.environ.get("HF_MODEL_REPO")
    hf_token = os.environ.get("HF_TOKEN")
    hf_filename = os.environ.get("HF_MODEL_FILENAME", DEPLOYMENT_CHECKPOINT_NAME)
    model_url = os.environ.get("MODEL_URL")

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if not hf_repo and "HF_MODEL_REPO" in st.secrets:
                hf_repo = st.secrets["HF_MODEL_REPO"]
            if not hf_token and "HF_TOKEN" in st.secrets:
                hf_token = st.secrets["HF_TOKEN"]
            if "HF_MODEL_FILENAME" in st.secrets:
                hf_filename = st.secrets["HF_MODEL_FILENAME"]
            if not model_url and "MODEL_URL" in st.secrets:
                model_url = st.secrets["MODEL_URL"]
    except Exception:
        pass

    # Default to user's verified public Hugging Face repository if none configured
    if not hf_repo and not model_url:
        hf_repo = "NavjotSingh987654321/navjot-singh-2005retinaai-weights"

    # 3. Attempt download from Hugging Face Hub if configured
    if hf_repo:
        try:
            print(f"[RetinaAI] Downloading model '{hf_filename}' from Hugging Face Hub ({hf_repo})...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_filename,
                token=hf_token or None,
            )
            shutil.copyfile(downloaded, str(ckpt_path))
            print(f"[RetinaAI] Checkpoint cached successfully at {ckpt_path}.")
            return ckpt_path
        except Exception as e:
            print(f"[RetinaAI] HuggingFace Hub client warning: {e}. Falling back to direct HTTPS download...")
            # Fallback to direct URL download from Hugging Face
            model_url = f"https://huggingface.co/{hf_repo}/resolve/main/{hf_filename}"

    # 4. Attempt download from direct URL if configured or via HF fallback
    if model_url:
        try:
            print(f"[RetinaAI] Downloading model from: {model_url}...")
            temp_path = ckpt_path.with_suffix(".tmp")
            headers = {"User-Agent": "RetinaAI-Deployment-Downloader/1.0"}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            req = urllib.request.Request(model_url, headers=headers)
            with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)

            if temp_path.stat().st_size < 10_000_000:
                temp_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file is too small to be a valid model checkpoint.")

            temp_path.replace(ckpt_path)
            print(f"[RetinaAI] Checkpoint cached successfully at {ckpt_path}.")
            return ckpt_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model checkpoint from URL ({model_url}): {e}"
            ) from e

    # 5. If neither exists, raise clear, actionable instructions
    if ckpt_path.exists():
        return ckpt_path

    raise FileNotFoundError(
        f"MANDATORY CHECKPOINT MISSING: Active model file '{ckpt_path.name}' was not found at {ckpt_path}.\n\n"
        "Deployment Setup Instructions for Streamlit Community Cloud:\n"
        "1. Host the model on Hugging Face Hub (free):\n"
        "   - Create a model repo at https://huggingface.co/new\n"
        "   - Upload 'retinaai_efficientnetv2s_deployment.pth'\n"
        "2. In Streamlit Cloud -> App Settings -> Secrets, add:\n"
        "   HF_MODEL_REPO = \"<your-hf-username>/<your-model-repo>\"\n"
        "   (or MODEL_URL = \"https://...direct-link-to-file...\")\n"
        "   HF_TOKEN = \"hf_...\" (only required if repo is private)\n"
        "3. Restart the Streamlit application."
    )


def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: torch.device = None,
) -> DRClassifier:
    """
    Load the trained DRClassifier from the active deployment checkpoint.
    Strictly loads retinaai_efficientnetv2s_deployment.pth with strict=True.
    No fallback allowed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure checkpoint is available locally on disk (auto-downloads if configured)
    ckpt_path = ensure_checkpoint_available(checkpoint_path)

    model = DRClassifier(pretrained=False)

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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
