"""
quality.py — Fundus Image Quality Assessment
==============================================
Heuristic-based quality gate to reject unusable images
BEFORE they enter the classification pipeline.

Checks:
  - Blur detection (Laplacian variance)
  - Brightness (mean pixel intensity)
  - Contrast (pixel intensity std)
  - Minimum resolution

Usage:
    from src.quality import assess_quality
    result = assess_quality(image)
    if not result["passed"]:
        print(result["reasons"])
"""

import cv2
import numpy as np
from typing import Dict, List, Any

from src.config import QUALITY


def compute_blur_score(image: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.

    Higher score = sharper image. Blurry fundus images cannot be
    reliably graded and should be rejected.

    Args:
        image: BGR image (H, W, 3).

    Returns:
        Laplacian variance (float). Lower = more blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute mean brightness of the image.

    Args:
        image: BGR image (H, W, 3).

    Returns:
        Mean pixel intensity (0-255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_contrast(image: np.ndarray) -> float:
    """
    Compute contrast as standard deviation of pixel intensities.

    Args:
        image: BGR image (H, W, 3).

    Returns:
        Std of pixel intensity (higher = more contrast).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def assess_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Full quality assessment for a fundus image.

    Runs all quality checks and returns a pass/fail verdict
    with individual metric scores and failure reasons.

    Args:
        image: BGR image (H, W, 3).

    Returns:
        Dictionary with:
            - passed (bool): Overall pass/fail
            - blur_score (float): Laplacian variance
            - brightness (float): Mean intensity
            - contrast (float): Intensity std
            - resolution (tuple): (height, width)
            - reasons (list): List of failure reason strings
    """
    h, w = image.shape[:2]
    blur_score = compute_blur_score(image)
    brightness = compute_brightness(image)
    contrast = compute_contrast(image)

    reasons: List[str] = []

    # Check blur
    if blur_score < QUALITY.min_blur_score:
        reasons.append(
            f"Image too blurry (score={blur_score:.1f}, "
            f"min={QUALITY.min_blur_score})"
        )

    # Check brightness
    if brightness < QUALITY.min_brightness:
        reasons.append(
            f"Image too dark (brightness={brightness:.1f}, "
            f"min={QUALITY.min_brightness})"
        )
    elif brightness > QUALITY.max_brightness:
        reasons.append(
            f"Image overexposed (brightness={brightness:.1f}, "
            f"max={QUALITY.max_brightness})"
        )

    # Check contrast
    if contrast < QUALITY.min_contrast:
        reasons.append(
            f"Insufficient contrast (contrast={contrast:.1f}, "
            f"min={QUALITY.min_contrast})"
        )

    # Check resolution
    if min(h, w) < QUALITY.min_image_size:
        reasons.append(
            f"Resolution too low ({w}x{h}, "
            f"min={QUALITY.min_image_size}px)"
        )

    return {
        "passed": len(reasons) == 0,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "resolution": (h, w),
        "reasons": reasons,
    }


def validate_fundus_domain(image: np.ndarray) -> Dict[str, Any]:
    """
    Lightweight, deterministic fundus domain validation gate.

    Distinguishes valid retinal fundus photography from non-fundus images
    (mango, fruits, selfies, screenshots, documents, random objects) using
    spectral, geometric, and color distribution signals BEFORE classification.

    Args:
        image: BGR image (H, W, 3) as numpy array.

    Returns:
        Dictionary with:
            - is_fundus (bool): True if valid fundus image domain, False otherwise
            - fundus_hue_ratio (float): Fraction of retinal tissue hue
            - yellow_ratio (float): Fraction of non-retinal yellow fruit content
            - green_ratio (float): Fraction of non-retinal green content
            - blue_ratio (float): Fraction of non-retinal blue/cyan content
            - white_ratio (float): Fraction of white background (document/screenshot)
            - reasons (list): List of domain rejection reason strings
    """
    if image is None or image.size == 0:
        return {'is_fundus': False, 'reasons': ['Empty or invalid image data']}

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)

    # Convert to HSV (OpenCV H is 0..180, S is 0..255, V is 0..255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # Mask non-black foreground pixels (brightness > 12)
    fg_mask = gray > 12
    fg_count = np.count_nonzero(fg_mask)

    if fg_count < (h * w * 0.05):
        return {'is_fundus': False, 'reasons': ['Image is completely dark or empty']}

    # Calculate average BGR on foreground
    mean_r = float(np.mean(r[fg_mask]))
    mean_g = float(np.mean(g[fg_mask]))
    mean_b = float(np.mean(b[fg_mask]))

    # Fundus Red/Orange/Red-Brown Hue Mask: H in [0, 22] or [155, 180], S > 20, V > 15
    fundus_hue_mask = fg_mask & (((h_chan <= 22) | (h_chan >= 155)) & (s_chan > 20) & (v_chan > 15))
    fundus_hue_ratio = float(np.count_nonzero(fundus_hue_mask) / fg_count)

    # Non-Fundus Yellow Hue (Mango, Banana, Yellow Fruit/Objects): H in [23, 38], S > 35
    yellow_mask = fg_mask & ((h_chan >= 23) & (h_chan <= 38) & (s_chan > 35))
    yellow_ratio = float(np.count_nonzero(yellow_mask) / fg_count)

    # Non-Fundus Green Hue (Vegetation, Leaves): H in [39, 85], S > 30
    green_mask = fg_mask & ((h_chan > 38) & (h_chan <= 85) & (s_chan > 30))
    green_ratio = float(np.count_nonzero(green_mask) / fg_count)

    # Non-Fundus Blue/Cyan Hue (Sky, Water, Blue objects): H in [86, 154], S > 30
    blue_mask = fg_mask & ((h_chan > 85) & (h_chan < 155) & (s_chan > 30))
    blue_ratio = float(np.count_nonzero(blue_mask) / fg_count)

    # Document/Screenshot White background mask
    white_mask = (r > 200) & (g > 200) & (b > 200)
    white_ratio = float(np.count_nonzero(white_mask) / (h * w))

    # Dark Border Ratio (Outer 5% margin dark pixel coverage)
    margin_h = max(1, int(h * 0.05))
    margin_w = max(1, int(w * 0.05))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin_h, :] = True
    border_mask[-margin_h:, :] = True
    border_mask[:, :margin_w] = True
    border_mask[:, -margin_w:] = True
    
    border_dark_ratio = float(np.count_nonzero((gray < 15) & border_mask) / np.count_nonzero(border_mask))

    # Geometric FOV Circularity Check
    contours, _ = cv2.findContours(fg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        perimeter = cv2.arcLength(largest_cnt, True)
        if perimeter > 0:
            circularity = float(4.0 * np.pi * area / (perimeter ** 2))

    # Structural Vessel / Optic-Disc Contrast Check in Green Channel
    green_fg = g[fg_mask]
    green_std = float(np.std(green_fg)) if len(green_fg) > 0 else 0.0

    reasons: List[str] = []

    # Check 1: Red dominance in BGR (Retinal tissue is red/orange dominant)
    if mean_r < (mean_b * 1.15):
        reasons.append(f"Color spectrum mismatch: Red channel not dominant over Blue (Mean Red={mean_r:.1f}, Blue={mean_b:.1f})")

    # Check 2: Yellow Mango/Fruit Check
    if yellow_ratio > 0.30 and fundus_hue_ratio < 0.40:
        reasons.append(f"Non-retinal color detected: High yellow fruit content ({yellow_ratio*100:.1f}%)")

    # Check 3: Green Vegetation/Object Check
    if green_ratio > 0.20:
        reasons.append(f"Non-retinal color detected: High green vegetation/object content ({green_ratio*100:.1f}%)")

    # Check 4: Blue/Cyan Sky/Object Check
    if blue_ratio > 0.20:
        reasons.append(f"Non-retinal color detected: High blue/cyan content ({blue_ratio*100:.1f}%)")

    # Check 5: Document / Screenshot Check
    if white_ratio > 0.35:
        reasons.append(f"Document/Screenshot detected: High white background ratio ({white_ratio*100:.1f}%)")

    # Check 6: Retinal Fundus Hue Coverage
    if fundus_hue_ratio < 0.15:
        reasons.append(f"Insufficient retinal fundus hue coverage ({fundus_hue_ratio*100:.1f}%, required >= 15%)")

    # Check 7: Geometric FOV Circularity (Reject non-circular full-bleed objects if zero dark border)
    if border_dark_ratio < 0.05 and circularity < 0.35:
        reasons.append(f"Geometric FOV mismatch: Non-circular foreground contour without dark border (Circularity={circularity:.2f})")

    # Check 8: Structural Blood Vessel / Tissue Contrast
    if green_std < 1.5:
        reasons.append(f"Structural sanity failed: Low green channel vessel contrast (std={green_std:.2f})")

    return {
        "is_fundus": len(reasons) == 0,
        "fundus_hue_ratio": fundus_hue_ratio,
        "yellow_ratio": yellow_ratio,
        "green_ratio": green_ratio,
        "blue_ratio": blue_ratio,
        "white_ratio": white_ratio,
        "border_dark_ratio": border_dark_ratio,
        "circularity": circularity,
        "green_std": green_std,
        "reasons": reasons,
    }


if __name__ == "__main__":
    print("Quality assessment module loaded successfully.")
    print(f"Blur threshold: {QUALITY.min_blur_score}")
    print(f"Brightness range: [{QUALITY.min_brightness}, {QUALITY.max_brightness}]")
    print(f"Min contrast: {QUALITY.min_contrast}")
    print(f"Min resolution: {QUALITY.min_image_size}px")

