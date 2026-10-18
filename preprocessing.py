"""
preprocessing.py — Retinal Fundus Image Preprocessing
======================================================
Implements the preprocessing pipeline for fundus images:
  1. Circle crop (remove black borders)
  2. Ben Graham's color normalization
  3. CLAHE contrast enhancement
  4. Resize to model input size
  5. Tensor normalization (ImageNet stats)

Usage:
    from src.preprocessing import preprocess_fundus, get_transforms
"""

import cv2
import numpy as np
from PIL import Image

from src.config import DATA


def circle_crop(image: np.ndarray) -> np.ndarray:
    """
    Crop the circular fundus region from a retinal image.

    Retinal fundus images typically have a circular field-of-view
    surrounded by black borders. This function detects and crops
    to the circular region.

    Args:
        image: BGR image (H, W, 3) as numpy array.

    Returns:
        Cropped BGR image containing only the fundus region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to find the bright fundus region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours and pick the largest one (the fundus)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add small padding
    pad = int(0.02 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    cropped = image[y:y + h, x:x + w]
    return cropped


def ben_graham_preprocessing(image: np.ndarray, sigma: int = 10) -> np.ndarray:
    """
    Ben Graham's preprocessing for fundus images.

    Subtracts a Gaussian-blurred version of the image to normalize
    color variations and enhance local contrast. Widely used in
    Kaggle DR competitions.

    Formula: result = α * image + β * gaussian_blur(image) + γ

    Args:
        image: BGR image (H, W, 3).
        sigma: Gaussian blur sigma (controls normalization scale).

    Returns:
        Color-normalized BGR image.
    """
    # Compute the Gaussian blur for background estimation
    ksize = 2 * sigma + 1
    gaussian = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    # Subtract local mean and rescale
    result = cv2.addWeighted(image, 4, gaussian, -4, 128)
    return result


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast in fundus images, making microaneurysms,
    hemorrhages, and exudates more visible.

    Args:
        image: BGR image (H, W, 3).
        clip_limit: CLAHE clip limit (higher = more contrast).
        grid_size: Tile grid size for local histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the L channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge back
    enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return result


def preprocess_fundus(image: np.ndarray, target_size: int = None) -> np.ndarray:
    """
    Full preprocessing pipeline for a single fundus image.

    Pipeline:
        1. Circle crop
        2. Ben Graham color normalization
        3. CLAHE contrast enhancement
        4. Resize to target_size × target_size

    Args:
        image: Raw BGR fundus image.
        target_size: Output size (default from config).

    Returns:
        Preprocessed BGR image of shape (target_size, target_size, 3).
    """
    if target_size is None:
        target_size = DATA.image_size

    # Step 1: Circle crop
    image = circle_crop(image)

    # Step 2: Resize to target_size (Accelerates Gaussian blur & CLAHE on CPU by 30x)
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Step 3: Ben Graham normalization
    image = ben_graham_preprocessing(image)

    # Step 4: CLAHE enhancement
    image = apply_clahe(image)

    return image


def get_train_transforms():
    """
    Get albumentations augmentation pipeline for training.

    Returns:
        albumentations.Compose transform pipeline.
    """
    # pyrefly: ignore [missing-import]
    import albumentations as A
    # pyrefly: ignore [missing-import]
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-30, 30),
            p=0.5,
        ),
        A.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.15, hue=0.03, p=0.4
        ),
        A.Resize(DATA.image_size, DATA.image_size),
        A.Normalize(mean=DATA.normalize_mean, std=DATA.normalize_std),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Get albumentations transform pipeline for validation/test.

    Returns:
        albumentations.Compose transform pipeline.
    """
    # pyrefly: ignore [missing-import]
    import albumentations as A
    # pyrefly: ignore [missing-import]
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(DATA.image_size, DATA.image_size),
        A.Normalize(mean=DATA.normalize_mean, std=DATA.normalize_std),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    # Quick test
    print("Preprocessing module loaded successfully.")
    print(f"Target image size: {DATA.image_size}x{DATA.image_size}")
    print(f"Normalization: mean={DATA.normalize_mean}, std={DATA.normalize_std}")
