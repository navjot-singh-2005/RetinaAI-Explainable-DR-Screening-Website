"""
src/provenance.py — Reproducibility, Determinism & Provenance Manifest Suite
============================================================================
Phase 10: Ensures end-to-end provenance tracking and deterministic reproducibility.

Captures:
  1. Git commit hash and working tree status.
  2. SHA-256 hash of active model weights (retinaai_efficientnetv2s_deployment.pth).
  3. SHA-256 hashes of all deduplicated dataset splits (train.csv, val.csv, test.csv).
  4. System, platform, architecture, and compute hardware metrics.
  5. Exact library versions (Python, PyTorch, Torchvision, Timm, Scikit-learn, etc.).
  6. Deterministic seeding utility (seed_everything).
"""

import os
import sys
import json
import random
import hashlib
import platform
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch

if Path(__file__).resolve().parent.name == "src":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, SYSTEM, MODEL, DECISION_POLICY


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all pseudo-random number generators across Python, NumPy, and PyTorch.
    Enforces deterministic algorithm execution where supported.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms with warning mode to prevent crash on non-deterministic ops
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file in streaming chunks."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return "file_not_found"

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info() -> Dict[str, str]:
    """Inspect Git repository commit, branch, and status with graceful fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(PATHS.project_root)
        ).decode().strip()
    except Exception:
        commit = "untracked_workspace"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(PATHS.project_root)
        ).decode().strip()
    except Exception:
        branch = "none"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=str(PATHS.project_root)
        ).decode().strip()
        is_clean = (len(status_out) == 0)
    except Exception:
        is_clean = False

    return {
        "commit": commit,
        "branch": branch,
        "is_clean": is_clean
    }


def get_split_hashes() -> Dict[str, str]:
    """Compute SHA-256 hashes of all deduplicated splits."""
    splits_dir = PATHS.data_splits.parent / "splits_dedup"
    hashes = {}
    for split_name in ["train.csv", "val.csv", "test.csv"]:
        split_path = splits_dir / split_name
        hashes[split_name] = get_file_sha256(split_path)
    return hashes


def get_library_versions() -> Dict[str, str]:
    """Gather exact runtime library versions."""
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__
    }

    optional_libs = [
        "torchvision", "timm", "albumentations", "cv2", "pandas",
        "scipy", "sklearn", "fastapi", "streamlit", "hypothesis"
    ]
    for lib in optional_libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[lib] = "not_installed"

    return versions


def get_device_info() -> Dict[str, Any]:
    """Gather compute hardware and environment specifications."""
    cuda_avail = torch.cuda.is_available()
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_avail,
    }
    if cuda_avail:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["cuda_device_name"] = "CPU"
    return info


def compute_config_hash() -> str:
    """Compute a deterministic hash of core system configuration parameters."""
    config_dict = {
        "model_version": SYSTEM.model_version,
        "policy_version": DECISION_POLICY.policy_version,
        "backbone": MODEL.backbone,
        "num_classes": MODEL.num_classes,
        "temperature": DECISION_POLICY.temperature,
        "operating_point": DECISION_POLICY.referable_operating_point,
        "ood_threshold": DECISION_POLICY.ood_energy_threshold,
        "uncertainty_threshold": DECISION_POLICY.uncertainty_refer_threshold,
    }
    dumped = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()


def generate_run_manifest(
    command: Optional[str] = None,
    seed: int = 42,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a complete, reproducible provenance manifest.
    
    Returns:
        Structured dictionary capturing exact runtime environment,
        code status, configuration hash, data split checksums, and weight checksum.
    """
    weights_path = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"
    weight_sha = get_file_sha256(weights_path)

    manifest = {
        "manifest_version": "1.0",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command or " ".join(sys.argv),
        "seed": seed,
        "git": get_git_info(),
        "config_hash": compute_config_hash(),
        "weight_sha256": weight_sha,
        "data_split_hashes": get_split_hashes(),
        "library_versions": get_library_versions(),
        "hardware_environment": get_device_info(),
        "system_version": {
            "app_name": SYSTEM.app_name,
            "version": SYSTEM.version,
            "model_version": SYSTEM.model_version,
            "policy_version": DECISION_POLICY.policy_version
        }
    }

    if additional_metadata:
        manifest["additional_metadata"] = additional_metadata

    return manifest


def save_run_manifest(manifest: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Write run manifest to designated JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    seed_everything(42)
    m = generate_run_manifest()
    print("Generated Provenance Manifest:")
    print(json.dumps(m, indent=2))
