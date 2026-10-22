"""
config.py — Centralized Configuration
======================================
All hyperparameters, paths, model settings, and clinical thresholds
are defined here. Import this module everywhere to ensure consistency.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ── Project Root ──────────────────────────────────────────────
# Resolves correctly whether config.py is in src/ or at the repository root
if Path(__file__).resolve().parent.name == "src":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PathConfig:
    """All filesystem paths."""
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_splits: Path = PROJECT_ROOT / "data" / "splits"
    model_checkpoints: Path = PROJECT_ROOT / "models" / "checkpoints"
    model_final: Path = PROJECT_ROOT / "models" / "final"
    output_predictions: Path = PROJECT_ROOT / "outputs" / "predictions"
    output_explanations: Path = PROJECT_ROOT / "outputs" / "explanations"
    output_graphs: Path = PROJECT_ROOT / "outputs" / "graphs"
    output_reports: Path = PROJECT_ROOT / "outputs" / "reports"


@dataclass
class DataConfig:
    """Dataset and preprocessing settings."""
    # APTOS 2019 settings
    csv_file: str = "train.csv"
    image_dir: str = "train_images"
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "No DR", "Mild", "Moderate", "Severe", "Proliferative"
    ])

    # Preprocessing
    image_size: int = 256
    crop_size: int = 256
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)  # ImageNet
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)   # ImageNet

    # Splits
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Dataloader
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    """Model architecture settings."""
    # Backbone
    backbone: str = "tf_efficientnetv2_s"
    pretrained: bool = True
    num_classes: int = 5
    dropout_rate: float = 0.3       # Used for both training and MC Dropout

    # MC Dropout inference
    mc_dropout_passes: int = 25     # Increased for robust uncertainty estimation

    # Test-Time Augmentation (TTA) — optional evaluation mode, default OFF
    enable_tta: bool = False
    tta_views: List[str] = field(default_factory=lambda: ["identity", "hflip", "rot90", "rot270"])


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 7
    min_delta: float = 0.001

    # Mixed precision
    use_amp: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 4
    monitor_metric: str = "val_qwk"  # Quadratic Weighted Kappa


@dataclass
class QualityConfig:
    """Image quality assessment thresholds."""
    min_blur_score: float = 50.0      # Laplacian variance threshold
    min_brightness: float = 30.0      # Mean pixel intensity minimum
    max_brightness: float = 220.0     # Mean pixel intensity maximum
    min_contrast: float = 20.0        # Std of pixel intensity
    min_image_size: int = 200         # Minimum dimension in pixels


@dataclass
class DecisionPolicyConfig:
    """
    Versioned clinical decision engine policy configuration (Policy v2.1).
    Centralizes all REFER / ROUTINE SCREENING / MANUAL REVIEW / REJECT rules,
    thresholds, and operating points.
    """
    policy_version: str = "v2.1-clinical-hardened"

    # Confidence thresholds
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.40

    # Uncertainty thresholds (predictive entropy)
    uncertainty_refer_threshold: float = 1.4497  # Empirical 95% TPR threshold from validation calibration

    # OOD detection (Higher energy = OOD. Empirical 95% TPR threshold from validation data)
    ood_energy_threshold: float = -1.6867  # Calibrated energy threshold

    # Referable DR definition: clinical standard Grade >= 2 (Moderate, Severe, Proliferative)
    referable_grades: List[int] = field(default_factory=lambda: [2, 3, 4])
    all_abnormal_grades: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # Calibrated probability operating point on P(referable DR, grade >= 2)
    # Empirical validation: Sensitivity 90.78% [86.83%, 94.39%], Specificity 75.93% [71.47%, 80.48%]
    referable_operating_point: float = 0.3080

    # Temperature scaling (learned post-training on validation set)
    temperature: float = 1.4239

    # Ordinal grading cutoffs (optional, disabled by default for direct softmax alignment)
    use_ordinal_staging: bool = False
    ordinal_thresholds: List[float] = field(default_factory=lambda: [0.5344, 1.3991, 2.0936, 3.8260])

    # Rejection and referral reason templates
    reason_non_fundus: str = "REJECT: Non-Fundus Input (Invalid Image Domain)"
    reason_poor_quality: str = "REJECT: Poor Image Quality"
    reason_ood: str = "REJECT: Out-of-Distribution/Invalid Image"
    reason_uncertain: str = "REFER: High Model Uncertainty"
    reason_referable_detected: str = "REFER: Referable DR Detected (Grade {grade})"
    reason_no_referable: str = "NO REFERRAL: No Referable DR Detected"
    reason_low_conf: str = "MANUAL REVIEW: Low Confidence"
    reason_possible_dr: str = "REVIEW/REFER: Possible DR Detected (Grade {grade})"
    reason_routine: str = "ROUTINE SCREENING: Likely No Referable DR"


# ScreeningConfig inherits from DecisionPolicyConfig for seamless backward-compatibility
@dataclass
class ScreeningConfig(DecisionPolicyConfig):
    """Clinical screening decision thresholds and policy."""
    pass


@dataclass
class AppConfig:
    """Streamlit app settings."""
    page_title: str = "DR Screening AI"
    page_icon: str = "👁️"
    max_upload_size_mb: int = 10
    supported_formats: List[str] = field(default_factory=lambda: [
        "png", "jpg", "jpeg", "tif", "tiff"
    ])


@dataclass
class SystemConfig:
    """System versioning settings."""
    app_name: str = "RetinaAI Automated DR Screening Engine"
    version: str = "2.1.0"
    model_version: str = "v2.0-EfficientNetV2-S"
    threshold_version: str = "calibrated-v1.4239"
    policy_version: str = "v2.1-clinical-hardened"


# ── Global Config Instance ────────────────────────────────────
# Import and use these throughout the project:
#   from src.config import PATHS, DATA, MODEL, TRAIN, QUALITY, SCREENING, DECISION_POLICY, APP, SYSTEM

PATHS = PathConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
QUALITY = QualityConfig()
DECISION_POLICY = DecisionPolicyConfig()
SCREENING = DECISION_POLICY
APP = AppConfig()
SYSTEM = SystemConfig()


def ensure_directories():
    """Create all project directories if they don't exist."""
    for field_name in PathConfig.__dataclass_fields__:
        path = getattr(PATHS, field_name)
        if isinstance(path, Path) and field_name != "project_root":
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PATHS.project_root}")
    print(f"Model: {MODEL.backbone}")
    print(f"Image size: {DATA.image_size}")
    print(f"Batch size: {DATA.batch_size}")
    print(f"MC Dropout passes: {MODEL.mc_dropout_passes}")
    print("All directories verified.")
