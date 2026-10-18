"""
explain.py — Grad-CAM Explainability
================================================================
Generates visual explanations for DR classification predictions
using Grad-CAM++ overlaid on fundus images.

Usage:
    from src.explain import generate_gradcam, overlay_heatmap
    heatmap, overlay = generate_gradcam(model, image_tensor, target_class=2)
"""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Optional, Tuple, List, Dict

from src.config import DATA


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Grad-CAM++ heatmap for a single image.

    Args:
        model: Trained DRClassifier model (in eval mode).
        input_tensor: Preprocessed image tensor, shape (1, 3, H, W).
        target_class: Class index to explain. None = predicted class.

    Returns:
        Grayscale normalized evidence map of shape (H, W), values in [0, 1].
    """
    model.eval()
    
    # Identify the target layer for GradCAM
    # EfficientNetV2-S uses conv_head as the terminal convolutional layer
    if hasattr(model.backbone, 'layer4'):
        target_layers = [model.backbone.layer4[-1]]
    elif hasattr(model.backbone, 'conv_head'):
        target_layers = [model.backbone.conv_head]
    elif hasattr(model.backbone, 'blocks'):
        target_layers = [model.backbone.blocks[-1]]
    else:
        # Fallback to the last layer before pooling
        target_layers = [list(model.backbone.children())[-2]]
        
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # Smooth the CAM heatmap (bicubic resize + light Gaussian blur)
    smoothed_cam = cv2.GaussianBlur(grayscale_cam, (7, 7), 1.5)
    
    # Normalize CAM to [0, 1]
    cam_min, cam_max = np.min(smoothed_cam), np.max(smoothed_cam)
    if cam_max > cam_min:
        norm_cam = (smoothed_cam - cam_min) / (cam_max - cam_min)
    else:
        norm_cam = smoothed_cam

    return norm_cam


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay evidence heatmap on the original fundus image.
    Masked to the circular fundus aperture with alpha = 0.35 opacity.

    Args:
        original_image: RGB image, shape (H, W, 3), values in [0, 1] or uint8 [0, 255].
        heatmap: Grayscale heatmap, shape (H, W), values in [0, 1].
        alpha: Transparency of the heatmap overlay (default: 0.35).
        colormap: OpenCV colormap (default: JET).

    Returns:
        RGB overlay image, shape (H, W, 3), values in uint8 [0, 255].
    """
    if original_image.dtype == np.uint8:
        img_float = original_image.astype(np.float32) / 255.0
        img_uint8 = original_image
    else:
        img_float = original_image.astype(np.float32)
        img_uint8 = (original_image * 255.0).clip(0, 255).astype(np.uint8)

    # Resize heatmap to match image if necessary
    if heatmap.shape[:2] != img_float.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_float.shape[1], img_float.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Fundus circular mask detection (threshold non-black region)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    mask_bool = mask > 0

    # Mask heatmap so black corners contain zero attention
    heatmap_masked = np.zeros_like(heatmap)
    heatmap_masked[mask_bool] = heatmap[mask_bool]

    # Normalize attention inside the mask only
    if np.any(mask_bool):
        c_min = np.min(heatmap_masked[mask_bool])
        c_max = np.max(heatmap_masked[mask_bool])
        if c_max > c_min:
            heatmap_masked[mask_bool] = (heatmap_masked[mask_bool] - c_min) / (c_max - c_min)

    # Generate overlay with grad-cam utility
    visualization = show_cam_on_image(img_float, heatmap_masked, use_rgb=True, colormap=colormap, image_weight=1.0 - alpha)

    # Zero out background corners outside fundus mask to keep corners clean black
    visualization[~mask_bool] = img_uint8[~mask_bool]

    return visualization


def generate_all_class_heatmaps(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Generate evidence heatmaps for ALL DR classes.
    """
    results = {}
    for class_idx, class_name in enumerate(DATA.class_names):
        heatmap = generate_gradcam(
            model, input_tensor, target_class=class_idx
        )
        overlay = overlay_heatmap(original_image, heatmap)
        results[class_name] = overlay

    return results


def generate_explanation_report(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    predicted_class: int,
    confidence: float,
) -> Dict:
    """
    Generate a complete explanation report for one image.
    """
    heatmap = generate_gradcam(
        model, input_tensor, target_class=predicted_class
    )
    overlay = overlay_heatmap(original_image, heatmap)

    # Compute Heatmap Spatial Entropy (Localization / Diffusion score)
    cam_flat = heatmap.flatten().astype(np.float64)
    cam_norm = cam_flat / (np.sum(cam_flat) + 1e-10)
    spatial_entropy = -np.sum(cam_norm * np.log(cam_norm + 1e-10)) / np.log(len(cam_flat))
    spatial_entropy = float(np.clip(spatial_entropy, 0.0, 1.0))

    # Attribution-Uncertainty Consistency Score
    # High consistency = focused heatmap for confident prediction or diffuse heatmap for high entropy
    norm_conf = float(confidence)
    consistency_score = float(1.0 - abs(spatial_entropy - (1.0 - norm_conf)))

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "class_name": DATA.class_names[predicted_class],
        "class_index": predicted_class,
        "confidence": confidence,
        "spatial_entropy": spatial_entropy,
        "attribution_consistency": consistency_score,
    }


# Backwards compatibility alias
generate_evidence_map = generate_gradcam

if __name__ == "__main__":
    print("Grad-CAM explainability module loaded successfully.")
