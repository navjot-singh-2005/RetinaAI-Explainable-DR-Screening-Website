"""
evaluate.py — Model Evaluation & Metrics
==========================================
Computes classification metrics including:
  - Accuracy, Precision, Recall, F1
  - Quadratic Weighted Kappa (clinical standard for DR)
  - Per-class metrics
  - Confusion matrix
  - AUC-ROC (one-vs-rest)

Usage:
    from src.evaluate import compute_metrics, full_evaluation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)

from src.config import PATHS, DATA


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Quadratic Weighted Kappa.

    QWK is the standard metric for DR grading as it penalizes
    predictions further from the true grade more heavily.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        QWK score in [-1, 1]. Higher is better.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C). Optional.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": compute_qwk(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # AUC (requires probabilities)
    if y_prob is not None and y_prob.shape[1] > 1:
        try:
            metrics["auc_ovr"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["auc_ovr"] = None

    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Per-class metrics
    for i, name in enumerate(DATA.class_names):
        binary_true = (y_true == i).astype(int)
        binary_pred = (y_pred == i).astype(int)
        metrics[f"sensitivity_{name}"] = recall_score(binary_true, binary_pred, zero_division=0)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=DATA.class_names,
        yticklabels=DATA.class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — DR Classification", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved: {save_path}")

    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot training history curves (loss, accuracy, QWK, LR).

    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', etc.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    if "val_acc" in history:
        axes[0, 1].plot(history["val_acc"], label="Val Accuracy", linewidth=2, color="green")
        axes[0, 1].set_title("Validation Accuracy", fontsize=13)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # QWK
    if "val_qwk" in history:
        axes[1, 0].plot(history["val_qwk"], label="Val QWK", linewidth=2, color="purple")
        axes[1, 0].set_title("Quadratic Weighted Kappa", fontsize=13)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    if "lr" in history:
        axes[1, 1].plot(history["lr"], label="LR", linewidth=2, color="orange")
        axes[1, 1].set_title("Learning Rate", fontsize=13)
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history saved: {save_path}")

    return fig


def full_evaluation(model, loader, device, save_dir: Path = None):
    """
    Run comprehensive evaluation: metrics + plots + report.

    Args:
        model: Trained model.
        loader: Test DataLoader.
        device: Target device.
        save_dir: Directory to save outputs (default: PATHS.output_reports).
    """
    import torch
    from torch.amp import autocast

    save_dir = save_dir or PATHS.output_reports
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with autocast(device_type=device.type, enabled=False):
                logits = model(images)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Print report
    print("\n" + "=" * 50)
    print("  EVALUATION REPORT")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<30} {value:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=DATA.class_names))

    # Save plots
    plot_confusion_matrix(all_labels, all_preds, save_dir / "confusion_matrix.png")
    print(f"\nResults saved to: {save_dir}")

    return metrics


if __name__ == "__main__":
    print("Evaluate module loaded successfully.")
    print(f"Class names: {DATA.class_names}")
