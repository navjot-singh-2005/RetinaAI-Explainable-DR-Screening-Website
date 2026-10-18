"""
train.py — 2-Stage Transfer Learning Training Loop
=====================================================
Complete training pipeline with:
  - Stage 1: Frozen backbone, train head only (3 epochs)
  - Stage 2: Unfrozen backbone with differential LR
  - Class-weighted CrossEntropy for imbalanced APTOS data
  - CosineAnnealing with warmup
  - Early stopping on Quadratic Weighted Kappa
  - Gradient clipping
  - Mixed precision (AMP)
  - Checkpointing

Usage:
    python -m src.train
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import PATHS, DATA, MODEL, TRAIN
from src.dataset import get_dataloaders, compute_class_weights, create_splits
from src.model import build_model
from src.evaluate import compute_metrics


# ── Stage Configuration ───────────────────────────────────────
STAGE1_EPOCHS = 3        # Head-only training
STAGE1_LR = 1e-3         # Higher LR for randomly initialized head
BACKBONE_LR = 1e-5       # Lower LR for pretrained backbone
HEAD_LR = 1e-4           # Higher LR for head in stage 2


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary with 'loss' and 'accuracy'.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(device_type=device.type, enabled=TRAIN.use_amp and device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Validate the model.

    Returns:
        Dictionary with 'loss', 'accuracy', 'qwk', 'all_preds', 'all_labels'.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=TRAIN.use_amp and device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    total = len(all_labels)
    epoch_loss = running_loss / total

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    metrics["all_probs"] = all_probs

    return metrics


def train(resume_from: str = None):
    """
    Full 2-stage training pipeline.

    Stage 1: Freeze backbone, train classification head only.
    Stage 2: Unfreeze backbone, use differential learning rates.
    """
    print("=" * 60)
    print("  DR Classification — 2-Stage Transfer Learning")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Class weights for imbalanced data
    train_csv = PATHS.data_splits / "train.csv"
    if train_csv.exists():
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        class_weights = compute_class_weights(train_df).to(device)
        print(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    else:
        class_weights = None

    # Model
    print("\nBuilding model...")
    model = build_model(device)

    # Loss — Weighted CrossEntropy to handle APTOS class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision scaler
    scaler = GradScaler(enabled=TRAIN.use_amp and device.type == 'cuda')

    # Ensure checkpoint dirs exist
    PATHS.model_checkpoints.mkdir(parents=True, exist_ok=True)
    PATHS.model_final.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_qwk": [], "lr": [], "stage": []}
    best_qwk = -1.0
    global_epoch = 0

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Freeze backbone, train head only
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STAGE 1: Training classification head only")
    print(f"  Backbone: FROZEN | Head LR: {STAGE1_LR}")
    print(f"  Epochs: {STAGE1_EPOCHS}")
    print("=" * 60)

    model.freeze_backbone(freeze=True)

    # Count trainable params (should be head only)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (head only): {trainable:,}")

    # Stage 1 optimizer — only head parameters
    optimizer_s1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR,
        weight_decay=TRAIN.weight_decay,
    )

    for epoch in range(STAGE1_EPOCHS):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_s1, scaler, device, global_epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, global_epoch)

        elapsed = time.time() - epoch_start
        print(
            f"[S1] Epoch {epoch+1}/{STAGE1_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_qwk"].append(val_metrics["qwk"])
        history["lr"].append(STAGE1_LR)
        history["stage"].append(1)

        # Save best model
        if val_metrics["qwk"] > best_qwk + TRAIN.min_delta:
            best_qwk = val_metrics["qwk"]
            best_path = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"
            torch.save({
                "epoch": global_epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "best_qwk": best_qwk,
                "val_metrics": {k: v for k, v in val_metrics.items()
                               if k not in ("all_preds", "all_labels", "all_probs")},
            }, best_path)
            print(f"  ✓ Best model saved (QWK: {best_qwk:.4f})")

        global_epoch += 1

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Unfreeze backbone, differential learning rates
    # ══════════════════════════════════════════════════════════
    stage2_epochs = TRAIN.epochs - STAGE1_EPOCHS

    print("\n" + "=" * 60)
    print("  STAGE 2: Full model fine-tuning (differential LR)")
    print(f"  Backbone LR: {BACKBONE_LR} | Head LR: {HEAD_LR}")
    print(f"  Epochs: {stage2_epochs}")
    print("=" * 60)

    model.freeze_backbone(freeze=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (full model): {trainable:,}")

    # Stage 2 optimizer — differential learning rates
    optimizer_s2 = AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=TRAIN.weight_decay)

    # Scheduler: Linear warmup → Cosine annealing
    warmup_scheduler = LinearLR(
        optimizer_s2, start_factor=0.1, total_iters=TRAIN.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer_s2, T_max=stage2_epochs - TRAIN.warmup_epochs, eta_min=TRAIN.min_lr
    )
    scheduler = SequentialLR(
        optimizer_s2,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[TRAIN.warmup_epochs],
    )

    patience_counter = 0

    for epoch in range(stage2_epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer_s2, scaler, device, global_epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, global_epoch)

        scheduler.step()
        current_lr = optimizer_s2.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start
        print(
            f"[S2] Epoch {epoch+1}/{stage2_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_qwk"].append(val_metrics["qwk"])
        history["lr"].append(current_lr)
        history["stage"].append(2)

        # Save best model
        if val_metrics["qwk"] > best_qwk + TRAIN.min_delta:
            best_qwk = val_metrics["qwk"]
            patience_counter = 0

            best_path = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"
            torch.save({
                "epoch": global_epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s2.state_dict(),
                "best_qwk": best_qwk,
                "val_metrics": {k: v for k, v in val_metrics.items()
                               if k not in ("all_preds", "all_labels", "all_probs")},
            }, best_path)
            print(f"  ✓ Best model saved (QWK: {best_qwk:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (global_epoch + 1) % TRAIN.save_every_n_epochs == 0:
            ckpt_path = PATHS.model_checkpoints / f"checkpoint_epoch_{global_epoch+1}.pth"
            torch.save({
                "epoch": global_epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s2.state_dict(),
                "best_qwk": best_qwk,
            }, ckpt_path)

        # Early stopping
        if patience_counter >= TRAIN.patience:
            print(f"\nEarly stopping at epoch {global_epoch+1} (patience={TRAIN.patience})")
            break

        global_epoch += 1

    # Final save if no best was saved
    best_path = PATHS.model_final / "retinaai_efficientnetv2s_deployment.pth"
    if not best_path.exists():
        torch.save({
            "epoch": global_epoch,
            "model_state_dict": model.state_dict(),
            "best_qwk": best_qwk,
            "val_metrics": {k: v for k, v in val_metrics.items()
                           if k not in ("all_preds", "all_labels", "all_probs")},
        }, best_path)
        print(f"  ✓ Model saved to {best_path}")

    # Save training history
    history_path = PATHS.output_reports / "training_history.json"
    PATHS.output_reports.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete. Best QWK: {best_qwk:.4f}")
    print(f"Best model: {best_path}")
    print(f"History: {history_path}")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train()
