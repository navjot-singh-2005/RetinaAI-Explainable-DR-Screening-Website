"""
predict.py — Inference & Screening Decision Engine
===================================================
Runs the complete inference pipeline on a single image and
applies clinical rules to generate a screening decision.

Pipeline:
  1. Quality Gate
  2. Preprocessing
  3. Feature Extraction & OOD Detection
  4. MC Dropout Classification
  5. Explainability (Grad-CAM++)
  6. Decision Engine

Usage:
    from src.predict import predict_pipeline
    report = predict_pipeline(image_path, model)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

from src.config import SCREENING, DECISION_POLICY, DecisionPolicyConfig, DATA, PATHS, MODEL
from src.quality import assess_quality, validate_fundus_domain
from src.preprocessing import preprocess_fundus
from src.uncertainty import mc_dropout_predict, compute_energy_score
from src.explain import generate_explanation_report
import src.utils as utils


def predict_with_tta(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Test-Time Augmentation (TTA) evaluation averaging across 4 canonical geometric views.
    Enabled via MODEL.enable_tta = True (default is False).
    """
    views = [
        tensor,
        torch.flip(tensor, dims=[-1]),            # horizontal flip
        torch.rot90(tensor, k=1, dims=[-2, -1]),  # 90 deg rotation
        torch.rot90(tensor, k=3, dims=[-2, -1]),  # 270 deg rotation
    ]
    all_mean_probs = []
    entropies = []
    epistemics = []
    aleatorics = []

    for v in views:
        res = mc_dropout_predict(model, v, device=device)
        all_mean_probs.append(res["mean_probs"])
        entropies.append(res["predictive_entropy"])
        epistemics.append(res["epistemic_uncertainty"])
        aleatorics.append(res["aleatoric_uncertainty"])

    avg_probs = np.mean(all_mean_probs, axis=0)
    avg_probs = avg_probs / np.sum(avg_probs)

    return {
        "mean_probs": avg_probs.tolist(),
        "predicted_class": int(np.argmax(avg_probs)),
        "predictive_entropy": float(np.mean(entropies)),
        "epistemic_uncertainty": float(np.mean(epistemics)),
        "aleatoric_uncertainty": float(np.mean(aleatorics)),
        "tta_enabled": True,
    }


def evaluate_clinical_decision(
    predicted_class: int,
    confidence: float,
    uncertainty: float,
    ood_score: float,
    quality_passed: bool,
    is_fundus: bool = True,
    referable_prob: Optional[float] = None,
    quality_report: Optional[Dict[str, Any]] = None,
    policy: DecisionPolicyConfig = DECISION_POLICY,
) -> Dict[str, Any]:
    """
    Hardened rule-based clinical decision engine (Policy v2.1).
    Evaluates clinical gating and thresholds, producing a recommended
    action, human-interpretable clinical rationale, and structured ungradable reason code.

    Returns:
        Dict with keys:
            - decision (str): Exact legacy clinical decision string (REFER, ROUTINE, REJECT)
            - decision_reason (str): Detailed clinical explanation
            - ungradable_code (str): 'NONE', 'DOMAIN_REJECT', 'QUALITY_FAIL', 'OOD_ENERGY'
            - policy_version (str): Active policy version (e.g. 'v2.1-clinical-hardened')
            - referable (bool): Whether case is clinically referable
    """
    # 1. Domain Gating (Non-fundus rejection)
    if not is_fundus:
        return {
            "decision": policy.reason_non_fundus,
            "decision_reason": "Image rejected: input photograph does not match retinal fundus characteristics.",
            "ungradable_code": "DOMAIN_REJECT",
            "policy_version": policy.policy_version,
            "referable": False,
        }

    # 2. Quality Gating (Laplacian blur / exposure failure)
    if not quality_passed:
        reasons_list = quality_report.get("reasons", []) if quality_report else []
        reasons_str = "; ".join(reasons_list) if reasons_list else "Laplacian blur variance or exposure below clinical thresholds"
        return {
            "decision": policy.reason_poor_quality,
            "decision_reason": f"Image ungradable: {reasons_str}. Retake recommended before grading.",
            "ungradable_code": "QUALITY_FAIL",
            "policy_version": policy.policy_version,
            "referable": False,
        }

    # 3. Out-of-Distribution Gating (Helmholtz Free Energy score)
    if ood_score > policy.ood_energy_threshold:
        return {
            "decision": policy.reason_ood,
            "decision_reason": f"Image ungradable / anomalous: Helmholtz free energy ({ood_score:.2f}) exceeds OOD threshold ({policy.ood_energy_threshold:.2f}).",
            "ungradable_code": "OOD_ENERGY",
            "policy_version": policy.policy_version,
            "referable": False,
        }

    # 4. Epistemic Uncertainty Gating (Predictive Entropy)
    if uncertainty > policy.uncertainty_refer_threshold:
        return {
            "decision": policy.reason_uncertain,
            "decision_reason": f"Specialist referral recommended: predictive entropy ({uncertainty:.3f} nats) exceeds safe screening threshold ({policy.uncertainty_refer_threshold:.3f} nats).",
            "ungradable_code": "NONE",
            "policy_version": policy.policy_version,
            "referable": True,
        }

    # 5. Operating Point & Severity Staging
    is_referable = (predicted_class in policy.referable_grades)
    if referable_prob is not None:
        # Check calibrated referable DR operating point (Sensitivity >= 90% threshold)
        is_referable = is_referable or (referable_prob >= policy.referable_operating_point)

    # Confident predictions
    if confidence >= policy.high_confidence_threshold:
        if is_referable:
            decision = policy.reason_referable_detected.format(grade=predicted_class)
            reason = f"High confidence ({confidence*100:.1f}%) detection of referable diabetic retinopathy (Grade {predicted_class}). Specialist ophthalmology referral required."
        else:
            decision = policy.reason_no_referable
            reason = f"High confidence ({confidence*100:.1f}%) non-referable status (Grade {predicted_class}). Routine annual diabetic screening recommended."
    # Low confidence / Ambiguous predictions
    elif confidence < policy.low_confidence_threshold:
        decision = policy.reason_low_conf
        reason = f"Borderline model confidence ({confidence*100:.1f}%). Secondary manual clinician review advised."
    # Intermediate confidence
    else:
        if is_referable:
            decision = policy.reason_possible_dr.format(grade=predicted_class)
            reason = f"Intermediate confidence ({confidence*100:.1f}%) with suspected referable DR (Grade {predicted_class}). Secondary review / referral advised."
        else:
            decision = policy.reason_routine
            reason = f"Intermediate confidence ({confidence*100:.1f}%) likely non-referable status (Grade {predicted_class}). Routine screening recommended."

    return {
        "decision": decision,
        "decision_reason": reason,
        "ungradable_code": "NONE",
        "policy_version": policy.policy_version,
        "referable": is_referable,
    }


def make_screening_decision(
    predicted_class: int,
    confidence: float,
    uncertainty: float,
    ood_score: float,
    quality_passed: bool,
    is_fundus: bool = True,
    referable_prob: Optional[float] = None,
) -> str:
    """
    Rule-based clinical decision engine.
    Maintains exact string return signature for 100% backward compatibility.
    """
    eval_res = evaluate_clinical_decision(
        predicted_class=predicted_class,
        confidence=confidence,
        uncertainty=uncertainty,
        ood_score=ood_score,
        quality_passed=quality_passed,
        is_fundus=is_fundus,
        referable_prob=referable_prob,
    )
    return eval_res["decision"]


def predict_pipeline(
    image_path: Union[str, Path],
    model: torch.nn.Module,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    End-to-end inference pipeline for a single fundus image.

    Args:
        image_path: Path to the image file.
        model: Trained DRClassifier.
        device: Target device.

    Returns:
        Dictionary containing all metrics, decisions, and images.
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Load image
    from src.utils import safe_imread
    raw_bgr = safe_imread(image_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # For visualization, we need original RGB
    original_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 1b. Fundus Domain Validation Gate (BEFORE Classification & Preprocessing)
    domain_report = validate_fundus_domain(raw_bgr)
    if not domain_report["is_fundus"]:
        decision_info = evaluate_clinical_decision(
            predicted_class=0,
            confidence=0.0,
            uncertainty=0.0,
            ood_score=0.0,
            quality_passed=False,
            is_fundus=False,
        )
        return {
            "status": "rejected",
            "is_fundus": False,
            "file": Path(image_path).name,
            "domain_validation": domain_report,
            "quality": {
                "passed": False,
                "score": 0.0,
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "resolution": (raw_bgr.shape[0], raw_bgr.shape[1]),
                "reasons": domain_report["reasons"],
            },
            "prediction": None,
            "uncertainty": None,
            "ood": None,
            "decision": decision_info["decision"],
            "decision_reason": decision_info["decision_reason"],
            "ungradable_code": decision_info["ungradable_code"],
            "policy_version": decision_info["policy_version"],
            "referable_prob": 0.0,
            "visuals": {
                "original": original_rgb,
                "preprocessed": original_rgb,
                "heatmap": None,
                "overlay": None,
            },
            "clinical_disclaimer": "RetinaAI is an automated clinical decision support screening aid and is NOT a standalone diagnostic device. All results must be reviewed by a certified eye care professional."
        }

    # 2. Quality Gate
    quality_report = assess_quality(raw_bgr)

    # Even if quality fails, we might still want to run the model for debugging
    # but the decision engine will reject it.

    # 3. Preprocessing
    preprocessed_bgr = preprocess_fundus(raw_bgr)
    preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor and apply ImageNet normalization
    # (Matches what albumentations does in dataset.py)
    tensor = preprocessed_rgb.astype(np.float32) / 255.0
    mean = np.array(DATA.normalize_mean, dtype=np.float32)
    std = np.array(DATA.normalize_std, dtype=np.float32)
    tensor = (tensor - mean) / std
    
    # Format for PyTorch: (1, C, H, W)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. OOD Detection (Energy Score)
    ood_score = compute_energy_score(model, tensor)

    # 5. MC Dropout Classification & Uncertainty (with optional TTA evaluation mode)
    if getattr(MODEL, "enable_tta", False):
        mc_results = predict_with_tta(model, tensor, device=device)
    else:
        mc_results = mc_dropout_predict(model, tensor, device=device)
    
    predicted_class = mc_results["predicted_class"]
    
    # Apply Temperature Scaling (learned post-training)
    temp = SCREENING.temperature
    raw_probs = np.array(mc_results["mean_probs"])
    # Apply temperature scaling to log probabilities
    log_probs = np.log(raw_probs + 1e-10)
    scaled_probs = np.exp(log_probs / temp) / np.sum(np.exp(log_probs / temp))
    calibrated_confidence = float(np.max(scaled_probs))
    predicted_class = int(np.argmax(scaled_probs))

    # Conformal prediction set (90% & 95% coverage guarantee)
    from scripts.calibrate import compute_conformal_sets
    conf_set_90 = compute_conformal_sets(scaled_probs[None, :], alpha=0.10)[0]
    conf_set_95 = compute_conformal_sets(scaled_probs[None, :], alpha=0.05)[0]

    # Normalize to [0, 1] for Grad-CAM overlay to match original behavior
    display_rgb = preprocessed_rgb.astype(np.float32) / 255.0

    # 6. Explainability (Grad-CAM++)
    explanation = generate_explanation_report(
        model, tensor, display_rgb, predicted_class, calibrated_confidence
    )

    # 7. Decision Engine
    referable_prob = float(np.sum(scaled_probs[2:]))
    decision_info = evaluate_clinical_decision(
        predicted_class=predicted_class,
        confidence=calibrated_confidence,
        uncertainty=mc_results["predictive_entropy"],
        ood_score=ood_score,
        quality_passed=quality_report["passed"],
        is_fundus=True,
        referable_prob=referable_prob,
        quality_report=quality_report,
    )
    decision = decision_info["decision"]

    # Build final report
    report = {
        "status": "success",
        "is_fundus": True,
        "domain_validation": domain_report,
        "file": Path(image_path).name,
        "quality": quality_report,
        "prediction": {
            "class_index": predicted_class,
            "class_name": DATA.class_names[predicted_class],
            "confidence": calibrated_confidence,
            "all_probabilities": scaled_probs.tolist(),
            "raw_probabilities": raw_probs.tolist(),
            "temperature_scaled": temp,
            "conformal_set_90": [DATA.class_names[i] for i in conf_set_90],
            "conformal_set_95": [DATA.class_names[i] for i in conf_set_95],
        },
        "uncertainty": {
            "entropy": mc_results["predictive_entropy"],
            "epistemic": mc_results["epistemic_uncertainty"],
            "aleatoric": mc_results["aleatoric_uncertainty"],
        },
        "ood": {
            "energy_score": ood_score,
        },
        "decision": decision,
        "decision_reason": decision_info["decision_reason"],
        "ungradable_code": decision_info["ungradable_code"],
        "policy_version": decision_info["policy_version"],
        "referable_prob": referable_prob,
        "visuals": {
            "original": original_rgb,
            "preprocessed": preprocessed_rgb,
            "heatmap": explanation["heatmap"],
            "overlay": explanation["overlay"],
        }
    }

    return report



if __name__ == "__main__":
    print("Predict module loaded successfully.")
    print("Referable grades:", SCREENING.referable_grades)
