"""
uncertainty.py — Uncertainty Estimation
=========================================
Monte Carlo Dropout uncertainty estimation for DR classification.

Performs T stochastic forward passes with dropout enabled at inference,
computing:
  - Mean prediction (averaged softmax probabilities)
  - Predictive entropy (total uncertainty)
  - Epistemic uncertainty (model uncertainty via mutual information)
  - Per-class variance

Usage:
    from src.uncertainty import mc_dropout_predict
    result = mc_dropout_predict(model, image_tensor)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from src.config import MODEL


def enable_mc_dropout(model: nn.Module):
    """
    Enable dropout layers while keeping everything else in eval mode.

    This is the key trick for MC Dropout: we want BatchNorm to use
    running statistics (eval mode) but Dropout to remain stochastic
    (train mode).

    Args:
        model: The neural network model (already in eval mode).
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Perform MC Dropout inference for uncertainty estimation.

    Runs T stochastic forward passes with dropout enabled,
    then computes uncertainty metrics from the distribution
    of predictions.

    Args:
        model: Trained DRClassifier.
        input_tensor: Shape (1, 3, H, W) or (B, 3, H, W).
        n_passes: Number of stochastic passes (default from config).
        device: Target device.

    Returns:
        Dictionary with:
            - mean_probs: Mean predicted probabilities (num_classes,)
            - predicted_class: Argmax of mean_probs
            - confidence: Max of mean_probs
            - predictive_entropy: Total uncertainty (scalar)
            - epistemic_uncertainty: Model uncertainty (scalar)
            - aleatoric_uncertainty: Data uncertainty (scalar)
            - per_class_variance: Variance per class (num_classes,)
            - all_probs: All T predictions (T, num_classes)
    """
    n_passes = n_passes or MODEL.mc_dropout_passes

    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    # Set model to eval, then re-enable dropout
    model.eval()
    enable_mc_dropout(model)

    # Collect T stochastic predictions
    all_probs = []
    with torch.no_grad():
        if hasattr(model, 'backbone') and hasattr(model, 'head') and input_tensor.shape[0] == 1:
            features = model.backbone(input_tensor)  # (1, 1280)
            features_rep = features.repeat(n_passes, 1)  # (T, 1280)
            logits = model.head(features_rep)  # (T, C)
            probs = F.softmax(logits, dim=1)  # (T, C)
            all_probs = probs.cpu()  # (T, C)
        else:
            for _ in range(n_passes):
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)  # (B, C)
                all_probs.append(probs.cpu())
            all_probs = torch.stack(all_probs, dim=0)
            if input_tensor.shape[0] == 1:
                all_probs = all_probs.squeeze(1)  # (T, C)

    # Mean prediction across T passes
    mean_probs = all_probs.mean(dim=0)  # (C,)

    # Predicted class and confidence
    confidence, predicted_class = mean_probs.max(dim=-1)

    # ── Uncertainty Metrics ────────────────────────────────────

    # 1. Predictive Entropy: H[y|x, D] = -Σ p̄ log p̄
    #    Total uncertainty = epistemic + aleatoric
    predictive_entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + 1e-10), dim=-1
    )

    # 2. Expected Entropy (aleatoric): E[H[y|x, θ]]
    #    Average entropy of individual predictions
    individual_entropies = -torch.sum(
        all_probs * torch.log(all_probs + 1e-10), dim=-1
    )  # (T,) or (T, B)
    aleatoric_uncertainty = individual_entropies.mean(dim=0)

    # 3. Epistemic Uncertainty (mutual information):
    #    I[y, θ|x, D] = H[y|x, D] - E[H[y|x, θ]]
    epistemic_uncertainty = predictive_entropy - aleatoric_uncertainty

    # 4. Per-class variance
    per_class_variance = all_probs.var(dim=0)  # (C,)

    return {
        "mean_probs": mean_probs.numpy(),
        "predicted_class": int(predicted_class.item()) if predicted_class.dim() == 0 else predicted_class.numpy(),
        "confidence": float(confidence.item()) if confidence.dim() == 0 else confidence.numpy(),
        "predictive_entropy": float(predictive_entropy.item()) if predictive_entropy.dim() == 0 else predictive_entropy.numpy(),
        "epistemic_uncertainty": float(epistemic_uncertainty.item()) if epistemic_uncertainty.dim() == 0 else epistemic_uncertainty.numpy(),
        "aleatoric_uncertainty": float(aleatoric_uncertainty.item()) if aleatoric_uncertainty.dim() == 0 else aleatoric_uncertainty.numpy(),
        "per_class_variance": per_class_variance.numpy(),
        "all_probs": all_probs.numpy(),
        "n_passes": n_passes,
    }


def compute_energy_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    temperature: float = 1.0,
    device: torch.device = None,
) -> float:
    """
    Compute energy-based OOD score from model logits.

    Energy(x) = -T * log(Σ exp(f_i(x) / T))

    Lower energy = more likely in-distribution.
    Higher energy = more likely out-of-distribution.

    Args:
        model: Trained classifier.
        input_tensor: Shape (1, 3, H, W).
        temperature: Temperature parameter.
        device: Target device.

    Returns:
        Energy score (float). Higher = more OOD.
    """
    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)  # (1, C)

    # Energy = -T * logsumexp(logits / T)
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
    return float(energy.item())


def calibrate_temperature(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """
    Learn optimal temperature for confidence calibration.

    Optimizes a single scalar T on the validation set to
    minimize negative log-likelihood. This is Temperature Scaling.

    Args:
        model: Trained model (frozen).
        val_loader: Validation DataLoader.
        device: Target device.
        max_iter: Maximum optimization iterations.
        lr: Learning rate for temperature optimization.

    Returns:
        Optimal temperature (float).
    """
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    # Optimize temperature using L-BFGS
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    optimal_temp = temperature.item()
    print(f"Optimal temperature: {optimal_temp:.4f}")
    return optimal_temp


if __name__ == "__main__":
    print("Uncertainty module loaded successfully.")
    print(f"MC Dropout passes: {MODEL.mc_dropout_passes}")
    print(f"Dropout rate: {MODEL.dropout_rate}")
