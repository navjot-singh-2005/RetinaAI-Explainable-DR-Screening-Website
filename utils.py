"""
utils.py — Helper Utilities
=============================
Miscellaneous helper functions for logging, seeding, etc.
"""

import os
import random
import cv2
import numpy as np
import torch
import logging
from src.config import DATA

def seed_everything(seed: int = None):
    """
    Seed all random number generators for reproducibility.
    """
    seed = seed or DATA.random_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name: str = "dr_screening") -> logging.Logger:
    """
    Set up basic logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def safe_imread(path_or_str, flags=cv2.IMREAD_COLOR):
    """Safely read an image from a path that may contain non-ASCII/Unicode characters on Windows."""
    import cv2
    path_str = str(path_or_str)
    if not os.path.exists(path_str):
        return None
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        if len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def safe_imwrite(path_or_str, image, ext=".png"):
    """Safely write an image to a path that may contain non-ASCII/Unicode characters on Windows."""
    import cv2
    path_str = str(path_or_str)
    os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
    ext = os.path.splitext(path_str)[1] or ext
    success, encoded = cv2.imencode(ext, image)
    if success:
        with open(path_str, "wb") as f:
            f.write(encoded.tobytes())
        return True
    return False


if __name__ == "__main__":
    print("Utils module loaded.")
