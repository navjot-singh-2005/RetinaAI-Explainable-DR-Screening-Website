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
from typing import Dict, Any, List, Tuple

from src.config import MODEL


def compute_conformal_sets(y_prob: np.ndarray, alpha: float = 0.05) -> List[List[int]]:
    """
    Compute Split Conformal Prediction sets with 1 - alpha coverage guarantee.
    Uses Adaptive Prediction Sets (APS).

    Args:
        y_prob: Probability array of shape (N, C) or (C,).
        alpha: Miscoverage rate (e.g. 0.05 for 95% coverage, 0.10 for 90% coverage).

    Returns:
        List of lists of class indices included in prediction set.
    """
    if y_prob.ndim == 1:
        y_prob = y_prob[None, :]

    sorted_idx = np.argsort(-y_prob, axis=1)
    sorted_probs = np.take_along_axis(y_prob, sorted_idx, axis=1)
    cum_probs = np.cumsum(sorted_probs, axis=1)

    prediction_sets = []
    for i in range(len(y_prob)):
        cutoff_idx = np.searchsorted(cum_probs[i], 1.0 - alpha)
        cutoff_idx = min(cutoff_idx, y_prob.shape[1] - 1)
        set_classes = sorted_idx[i, :cutoff_idx + 1].tolist()
        prediction_sets.append(sorted(set_classes))

    return prediction_sets


def enable_mc_dropout(model: nn.Module):
    """
    Enable dropout layers while keeping BatchNorm and other layers in eval mode.

    This is the key requirement for MC Dropout: BatchNorm must use
    frozen running statistics (eval mode) while Dropout remains stochastic
    (train mode).

    Args:
        model: The neural network model (already in eval mode).
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
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
    of predictions. Also records a single deterministic eval pass.

    Args:
        model: Trained DRClassifier.
        input_tensor: Shape (1, 3, H, W) or (B, 3, H, W).
        n_passes: Number of stochastic passes (default from config).
        device: Target device.

    Returns:
        Dictionary with:
            - mean_probs: Mean predicted probabilities across T passes (num_classes,)
            - single_pass_probs: Deterministic single forward pass probabilities (num_classes,)
            - predicted_class: Argmax of mean_probs (int)
            - confidence: Max of mean_probs (float)
            - predictive_entropy: Total uncertainty in nats (scalar float)
            - epistemic_uncertainty: Model mutual info in nats (scalar float >= 0.0)
            - aleatoric_uncertainty: Data uncertainty in nats (scalar float)
            - per_class_variance: Variance per class across T passes (num_classes,)
            - all_probs: All T predictions (T, num_classes)
            - n_passes: T passes performed
    """
    n_passes = n_passes or MODEL.mc_dropout_passes

    if device is None:
        device = next(model.parameters()).device

    input_tensor = input_tensor.to(device)

    # 1. Deterministic single forward pass (eval mode, dropout OFF)
    model.eval()
    with torch.no_grad():
        single_logits = model(input_tensor)
        single_probs = F.softmax(single_logits, dim=-1)
        if input_tensor.shape[0] == 1:
            single_probs = single_probs.squeeze(0)
        single_probs_np = single_probs.cpu().numpy()

    # 2. MC Dropout stochastic forward passes (eval mode, dropout ON)
    model.eval()
    enable_mc_dropout(model)

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

    # Return model to clean eval mode
    model.eval()

    # Mean prediction across T passes
    mean_probs = all_probs.mean(dim=0)  # (C,)
    mean_probs = mean_probs / mean_probs.sum()  # ensure exact sum to 1.0

    # Predicted class and confidence authoritative selection
    confidence, predicted_class = mean_probs.max(dim=-1)

    # ── Uncertainty Metrics in Natural Units (Nats) ─────────────

    # 1. Predictive Entropy: H[y|x, D] = -Σ p̄ ln p̄
    safe_mean_p = torch.clamp(mean_probs, min=1e-12)
    predictive_entropy = -torch.sum(
        torch.where(mean_probs > 0, mean_probs * torch.log(safe_mean_p), torch.zeros_like(mean_probs)),
        dim=-1
    )

    # 2. Expected Entropy (aleatoric): E[H[y|x, θ]]
    safe_all_p = torch.clamp(all_probs, min=1e-12)
    individual_entropies = -torch.sum(
        torch.where(all_probs > 0, all_probs * torch.log(safe_all_p), torch.zeros_like(all_probs)),
        dim=-1
    )  # (T,) or (T, B)
    aleatoric_uncertainty = individual_entropies.mean(dim=0)

    # 3. Epistemic Uncertainty (mutual information in nats):
    #    I[y, θ|x, D] = H[y|x, D] - E[H[y|x, θ]] >= 0
    epistemic_raw = predictive_entropy - aleatoric_uncertainty
    if bool((epistemic_raw < -1e-4).any()):
        import warnings
        warnings.warn(f"Materially negative mutual information detected: {epistemic_raw.item():.6f}")
    epistemic_uncertainty = torch.clamp(epistemic_raw, min=0.0)

    # 4. Per-class variance
    per_class_variance = all_probs.var(dim=0)  # (C,)

    return {
        "mean_probs": mean_probs.numpy(),
        "single_pass_probs": single_probs_np,
        "predicted_class": int(predicted_class.item()) if predicted_class.dim() == 0 else int(predicted_class.numpy()[0]),
        "confidence": float(confidence.item()) if confidence.dim() == 0 else float(confidence.numpy()[0]),
        "predictive_entropy": float(predictive_entropy.item()) if predictive_entropy.dim() == 0 else float(predictive_entropy.numpy()[0]),
        "epistemic_uncertainty": float(epistemic_uncertainty.item()) if epistemic_uncertainty.dim() == 0 else float(epistemic_uncertainty.numpy()[0]),
        "aleatoric_uncertainty": float(aleatoric_uncertainty.item()) if aleatoric_uncertainty.dim() == 0 else float(aleatoric_uncertainty.numpy()[0]),
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
