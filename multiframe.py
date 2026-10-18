"""
multiframe.py — Smartphone Multi-Frame Fundus Analysis
=======================================================
Extracts frames from video/burst uploads, runs quality filtering,
selects the best frames, performs EfficientNetV2-S inference on each,
and fuses probabilities into a single screening result.

Pipeline:
  Video/Burst → Extract → Quality Gate → Select Best → Predict → Fuse

Usage:
    from src.multiframe import run_multiframe_pipeline, extract_frames_from_video
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Any, Tuple

from src.predict import predict_pipeline, make_screening_decision
from src.quality import assess_quality
from src.config import DATA

MAX_ANALYZED_FRAMES = 5


def extract_frames_from_video(video_bytes: bytes, max_frames: int = 10) -> List[Dict[str, Any]]:
    """
    Extract evenly spaced frames from a video byte stream.

    Returns a list of dicts, each containing:
        - "frame_index": the actual frame index in the video
        - "timestamp": timestamp in seconds
        - "image_bytes": PNG-encoded frame bytes
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    cap = cv2.VideoCapture(tmp_path)

    if not cap.isOpened():
        os.remove(tmp_path)
        raise ValueError("Failed to open video file for frame extraction.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames == 0:
        os.remove(tmp_path)
        raise ValueError("Video contains no frames.")

    # Calculate step to get max_frames evenly spaced
    step = max(1, total_frames // max_frames)

    count = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret or extracted >= max_frames:
            break

        if count % step == 0:
            success, buffer = cv2.imencode(".png", frame)
            if success:
                frames.append({
                    "frame_index": count,
                    "timestamp": round(count / fps, 3),
                    "image_bytes": buffer.tobytes(),
                })
                extracted += 1
        count += 1

    cap.release()
    os.remove(tmp_path)
    return frames


def extract_frames_from_images(bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    """
    Wrap raw image bytes (from burst upload) into the same metadata
    structure as video frames.
    """
    frames = []
    for i, b in enumerate(bytes_list):
        frames.append({
            "frame_index": i,
            "timestamp": 0.0,  # no timestamp for still images
            "image_bytes": b,
        })
    return frames


def run_multiframe_pipeline(
    bytes_list: List[bytes],
    model,
    device,
    *,
    is_video: bool = False,
    max_analyzed: int = MAX_ANALYZED_FRAMES,
    fusion_method: str = "mean",
) -> Dict[str, Any]:
    """
    Full multiframe pipeline with rich per-frame metadata.

    Steps:
        1. Extract frames (already done externally for video; wrap for images)
        2. Run quality assessment on every frame
        3. Reject poor-quality frames
        4. Select the best usable frames (by quality score, up to max_analyzed)
        5. Run DR model inference on selected frames only
        6. Fuse probabilities across analyzed frames
        7. Return comprehensive report with telemetry

    Returns a report dict compatible with the existing app.py display code.
    """
    if not bytes_list:
        raise ValueError("No frames provided to multiframe pipeline.")

    # ── Step 1: Build frame metadata list ──────────────────────────
    raw_frames = extract_frames_from_images(bytes_list)
    total_captured = len(raw_frames)

    # ── Step 2 & 3: Quality assessment on every frame ──────────────
    all_frame_meta = []  # metadata for ALL frames (including rejected)
    usable_frames = []   # frames that pass quality

    for frame_info in raw_frames:
        # Decode image bytes for quality assessment
        img_array = np.frombuffer(frame_info["image_bytes"], dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            all_frame_meta.append({
                "frame_index": frame_info["frame_index"],
                "timestamp": frame_info["timestamp"],
                "quality_score": 0.0,
                "quality_status": "Rejected",
                "quality_reasons": ["Failed to decode image"],
                "selected": False,
                "analyzed": False,
                "prediction": None,
            })
            continue

        qual = assess_quality(img_bgr)
        quality_score = qual["blur_score"]

        meta = {
            "frame_index": frame_info["frame_index"],
            "timestamp": frame_info["timestamp"],
            "quality_score": quality_score,
            "quality_brightness": qual["brightness"],
            "quality_contrast": qual["contrast"],
            "quality_passed": qual["passed"],
            "quality_reasons": qual["reasons"],
            "quality_status": "Usable" if qual["passed"] else "Rejected",
            "selected": False,
            "analyzed": False,
            "prediction": None,
        }
        all_frame_meta.append(meta)

        if qual["passed"]:
            usable_frames.append((meta, frame_info))

    usable_count = len(usable_frames)
    rejected_count = total_captured - usable_count

    # ── Step 4: Select best frames by quality score ────────────────
    # Sort usable frames by quality score descending, pick top N
    usable_frames.sort(key=lambda x: x[0]["quality_score"], reverse=True)
    selected_frames = usable_frames[:max_analyzed]

    # If no usable frames, fallback to the single best-quality frame
    if not selected_frames:
        # Sort ALL frames by quality and take the best one even if it failed
        best_idx = 0
        best_score = -1
        for i, meta in enumerate(all_frame_meta):
            if meta["quality_score"] > best_score:
                best_score = meta["quality_score"]
                best_idx = i
        # Use this frame anyway
        fallback_info = raw_frames[best_idx]
        all_frame_meta[best_idx]["selected"] = True
        selected_frames = [(all_frame_meta[best_idx], fallback_info)]

    # Mark selected frames
    selected_indices = set()
    for meta, _ in selected_frames:
        meta["selected"] = True
        selected_indices.add(meta["frame_index"])

    selected_count = len(selected_frames)

    # ── Step 5: Run model inference on selected frames only ────────
    per_frame_reports = []
    analyzed_count = 0

    for meta, frame_info in selected_frames:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(frame_info["image_bytes"])
            tmp_path = tmp.name

        try:
            report = predict_pipeline(tmp_path, model, device=device)
            meta["analyzed"] = True
            meta["quality_status"] = "Analyzed"
            meta["prediction"] = {
                "class_index": report["prediction"]["class_index"],
                "class_name": report["prediction"]["class_name"],
                "confidence": report["prediction"]["confidence"],
                "all_probabilities": report["prediction"]["all_probabilities"],
            }
            per_frame_reports.append({
                "frame_index": meta["frame_index"],
                "timestamp": meta["timestamp"],
                "quality_score": meta["quality_score"],
                "report": report,
                "prediction": report["prediction"],
                "uncertainty": report["uncertainty"],
                "ood": report["ood"],
                "quality": report["quality"],
            })
            analyzed_count += 1
        except Exception as e:
            print(f"Frame {meta['frame_index']} inference error: {e}")
            meta["quality_status"] = "Error"
        finally:
            os.remove(tmp_path)

    if analyzed_count == 0:
        raise ValueError("No suitable fundus frames available for analysis.")

    # ── Step 6: Fuse probabilities across analyzed frames ───────────
    all_probs = np.array([np.array(pfr["prediction"]["all_probabilities"]) for pfr in per_frame_reports]) # (K, C)
    quality_scores = np.array([pfr["quality_score"] for pfr in per_frame_reports]) # (K,)

    if fusion_method == "quality_weighted" and len(quality_scores) > 0 and np.sum(quality_scores) > 0:
        weights = quality_scores / np.sum(quality_scores)
        fused_probs = np.sum(all_probs * weights[:, None], axis=0)
    elif fusion_method == "median":
        med_probs = np.median(all_probs, axis=0)
        fused_probs = med_probs / np.sum(med_probs)
        weights = np.ones(len(per_frame_reports)) / len(per_frame_reports)
    elif fusion_method == "max_pooling":
        max_p = np.max(all_probs, axis=0)
        fused_probs = max_p / np.sum(max_p)
        weights = np.ones(len(per_frame_reports)) / len(per_frame_reports)
    else:  # "mean" (equal-weighted default)
        fused_probs = np.mean(all_probs, axis=0)
        weights = np.ones(len(per_frame_reports)) / len(per_frame_reports)

    ensemble_class_idx = int(np.argmax(fused_probs))
    ensemble_conf = float(np.max(fused_probs))

    # Pick best frame (highest quality among analyzed) for visuals
    best_frame_report = max(per_frame_reports, key=lambda r: r["quality_score"])
    best_frame_idx = best_frame_report["frame_index"]
    exemplar_report = best_frame_report["report"]

    # Average uncertainty across analyzed frames
    avg_entropy = float(np.mean([pfr["uncertainty"]["entropy"] for pfr in per_frame_reports]))
    avg_epistemic = float(np.mean([pfr["uncertainty"]["epistemic"] for pfr in per_frame_reports]))
    avg_aleatoric = float(np.mean([pfr["uncertainty"]["aleatoric"] for pfr in per_frame_reports]))
    avg_ood = float(np.mean([pfr["ood"]["energy_score"] for pfr in per_frame_reports]))

    # Inter-frame disagreement (std of per-frame confidences)
    per_frame_confs = [pfr["prediction"]["confidence"] for pfr in per_frame_reports]
    inter_frame_disagreement = float(np.std(per_frame_confs)) if len(per_frame_confs) > 1 else 0.0

    # Re-evaluate the decision based on ensemble
    decision = make_screening_decision(
        predicted_class=ensemble_class_idx,
        confidence=ensemble_conf,
        uncertainty=avg_entropy,
        ood_score=avg_ood,
        quality_passed=True,
    )

    # ── Step 7: Build comprehensive report ─────────────────────────
    return {
        "status": "success",
        "mode": "multiframe",
        "file": "Smartphone Multi-Frame Burst",
        "prediction": {
            "class_index": ensemble_class_idx,
            "class_name": DATA.class_names[ensemble_class_idx],
            "confidence": ensemble_conf,
            "all_probabilities": fused_probs.tolist(),
            "fusion_method": fusion_method,
        },
        "uncertainty": {
            "entropy": avg_entropy,
            "epistemic": avg_epistemic,
            "aleatoric": avg_aleatoric,
        },
        "ood": {
            "energy_score": avg_ood,
        },
        "decision": decision,
        "quality": exemplar_report["quality"],
        "visuals": exemplar_report["visuals"],
        "exemplar_index": best_frame_idx,
        "telemetry": {
            "total_captured": total_captured,
            "usable_frames": usable_count,
            "rejected_frames": rejected_count,
            "selected_frames": selected_count,
            "analyzed_frames": analyzed_count,
            "selected_scores": [pfr["quality_score"] for pfr in per_frame_reports],
            "selected_frame_indices": [pfr["frame_index"] for pfr in per_frame_reports],
            "inter_frame_disagreement": inter_frame_disagreement,
            "fusion_method": fusion_method,
        },
        "per_frame_reports": per_frame_reports,
        "all_frame_meta": all_frame_meta,
        "weights": weights.tolist(),
    }
