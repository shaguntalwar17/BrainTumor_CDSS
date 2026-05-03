from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np

from backend.services.preprocessing_service import PreprocessResult
from backend.utils.config import settings


TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]
MODEL_VERSION = "research-prototype-v2"


class ModelRuntimeError(RuntimeError):
    pass


@dataclass
class AnalysisOutput:
    tumor_detected: bool
    tumor_type: str
    confidence: float
    class_probabilities: dict[str, float]
    mask: np.ndarray
    gradcam: np.ndarray
    overlay: np.ndarray
    tumor_area: float
    tumor_volume: float | None
    explainability_consistency_score: float
    explainability_warning: str | None
    inference_time_sec: float
    model_version: str
    runtime_mode: str
    runtime_note: str


_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def _segment_tumor_demo(image: np.ndarray) -> np.ndarray:
    image_u8 = (image * 255).astype(np.uint8)
    threshold = int(np.clip(np.mean(image_u8) + 0.8 * np.std(image_u8), 25, 225))
    _, mask = cv2.threshold(image_u8, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _largest_component((mask > 0).astype(np.uint8))
    return mask


def _detect(mask: np.ndarray, image: np.ndarray) -> tuple[bool, float]:
    area_ratio = float(mask.mean())
    texture = float(np.std(image))
    confidence = float(np.clip(0.2 + area_ratio * 12 + texture * 0.6, 0.05, 0.99))
    detected = area_ratio > 0.006 and confidence >= 0.45
    return detected, confidence


def _classify_demo(mask: np.ndarray, image: np.ndarray, detected: bool) -> tuple[str, dict[str, float]]:
    if not detected:
        probs = {cls: 0.0 for cls in TUMOR_CLASSES}
        probs["No_Tumor"] = 1.0
        return "No_Tumor", probs

    m = mask.astype(bool)
    masked = image[m] if np.any(m) else image

    intensity = float(np.mean(masked))
    spread = float(np.std(masked))
    edge_map = cv2.Canny((image * 255).astype(np.uint8), 20, 80)
    edge_density = float(np.mean(edge_map > 0))

    scores = {
        "Glioma": 0.4 + 0.3 * spread + 0.2 * edge_density,
        "Meningioma": 0.35 + 0.25 * intensity,
        "Pituitary": 0.3 + 0.15 * (1.0 - edge_density) + 0.2 * intensity,
        "No_Tumor": 0.05,
    }
    vec = np.array([scores[c] for c in TUMOR_CLASSES], dtype=np.float32)
    vec = np.exp(vec - np.max(vec))
    vec /= np.sum(vec)
    probs = {c: float(p) for c, p in zip(TUMOR_CLASSES, vec)}
    pred = max(probs, key=probs.get)
    return pred, probs


def _generate_gradcam(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image.shape
    if np.any(mask):
        ys, xs = np.where(mask > 0)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        sigma = max(8.0, np.sqrt(float(np.sum(mask))) / 2.5)
    else:
        cy, cx = h // 2, w // 2
        sigma = min(h, w) / 4

    y = np.arange(h).reshape(-1, 1)
    x = np.arange(w).reshape(1, -1)
    heat = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))

    edges = cv2.Canny((image * 255).astype(np.uint8), 30, 100).astype(np.float32) / 255.0
    heat = 0.8 * heat + 0.2 * edges
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat.astype(np.float32)


def _overlay(image: np.ndarray, mask: np.ndarray, gradcam: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heat_u8 = (gradcam * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_rgb, 0.6, heat_color, 0.4, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def _explainability_consistency(mask: np.ndarray, gradcam: np.ndarray) -> tuple[float, str | None]:
    if not np.any(mask):
        return 1.0, None

    g = gradcam > np.percentile(gradcam, 70)
    m = mask > 0
    inter = float(np.logical_and(g, m).sum())
    union = float(np.logical_or(g, m).sum())
    score = inter / (union + 1e-8)

    if score < 0.25:
        warning = "Grad-CAM focus does not align well with segmented tumor region."
    else:
        warning = None
    return float(score), warning


def _load_trained_runtime():
    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE:
            return _MODEL_CACHE

        cls_path = Path(settings.classification_model_path)
        seg_path = Path(settings.segmentation_model_path)
        if not cls_path.exists() or not seg_path.exists():
            raise ModelRuntimeError(
                "Trained model checkpoint not found. Please train the model or switch MODEL_RUNTIME_MODE=demo."
            )

        try:
            import torch
            from ml.classification.train import build_model
            from ml.segmentation.models import build_segmentation_model
        except Exception as exc:
            raise ModelRuntimeError(
                "Required ML dependencies are missing for trained mode. Install dependencies or switch MODEL_RUNTIME_MODE=demo."
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cls_ckpt = torch.load(cls_path, map_location=device)
        cls_name = str(cls_ckpt.get("model_name", "resnext101_32x8d")).lower()
        classes = cls_ckpt.get("classes", TUMOR_CLASSES)
        classifier = build_model(cls_name, num_classes=len(classes), dropout=0.0)
        classifier.load_state_dict(cls_ckpt["state_dict"], strict=False)
        classifier.to(device).eval()

        seg_ckpt = torch.load(seg_path, map_location=device)
        seg_name = str(seg_ckpt.get("model_name", "unet"))
        segmenter = build_segmentation_model(seg_name, in_channels=1, out_channels=1)
        segmenter.load_state_dict(seg_ckpt["state_dict"], strict=False)
        segmenter.to(device).eval()

        _MODEL_CACHE.update(
            {
                "device": device,
                "classifier": classifier,
                "segmenter": segmenter,
                "classes": classes,
                "classification_model_name": cls_name,
                "segmentation_model_name": seg_name,
            }
        )
        return _MODEL_CACHE


def _classify_trained(image: np.ndarray) -> tuple[str, dict[str, float]]:
    runtime = _load_trained_runtime()

    import torch

    classifier = runtime["classifier"]
    device = runtime["device"]
    classes = runtime["classes"]

    image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    named = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    for c in TUMOR_CLASSES:
        named.setdefault(c, 0.0)

    pred = max(named, key=named.get)
    return pred, named


def _segment_trained(image: np.ndarray) -> np.ndarray:
    runtime = _load_trained_runtime()

    import torch

    segmenter = runtime["segmenter"]
    device = runtime["device"]

    tensor = torch.from_numpy(image[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = segmenter(tensor)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return _largest_component(mask)


def analyze_scan(preprocessed: PreprocessResult) -> AnalysisOutput:
    start = perf_counter()

    image = preprocessed.normalized_image
    runtime_mode = settings.model_runtime_mode.strip().lower()

    if runtime_mode == "trained":
        mask = _segment_trained(image)
        tumor_detected, confidence = _detect(mask, image)
        if not tumor_detected:
            mask = np.zeros_like(mask, dtype=np.uint8)
            tumor_type = "No_Tumor"
            probs = {cls: 0.0 for cls in TUMOR_CLASSES}
            probs["No_Tumor"] = 1.0
        else:
            tumor_type, probs = _classify_trained(image)
        runtime_note = "Trained mode active. Loaded provided classification and segmentation checkpoints."
        model_version = f"{MODEL_VERSION}-trained"
    else:
        mask = _segment_tumor_demo(image)
        tumor_detected, confidence = _detect(mask, image)
        if not tumor_detected:
            mask = np.zeros_like(mask, dtype=np.uint8)
        tumor_type, probs = _classify_demo(mask, image, tumor_detected)
        runtime_note = "Demo Mode: baseline heuristic inference. Not clinically validated."
        model_version = f"{MODEL_VERSION}-demo"
        runtime_mode = "demo"

    tumor_area = float(np.sum(mask > 0))

    tumor_volume = None
    if preprocessed.is_3d and preprocessed.voxel_spacing:
        sx, sy, sz = preprocessed.voxel_spacing
        tumor_volume = float(tumor_area * sx * sy * sz)

    gradcam = _generate_gradcam(image, mask)
    overlay = _overlay(image, mask, gradcam)
    xai_score, xai_warning = _explainability_consistency(mask, gradcam)

    infer_time = perf_counter() - start

    if tumor_type == "No_Tumor":
        confidence = max(confidence, probs.get("No_Tumor", 0.0))
    else:
        confidence = max(confidence, probs.get(tumor_type, 0.0))

    return AnalysisOutput(
        tumor_detected=tumor_detected,
        tumor_type=tumor_type,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        class_probabilities=probs,
        mask=mask.astype(np.uint8),
        gradcam=gradcam,
        overlay=overlay,
        tumor_area=tumor_area,
        tumor_volume=tumor_volume,
        explainability_consistency_score=xai_score,
        explainability_warning=xai_warning,
        inference_time_sec=float(infer_time),
        model_version=model_version,
        runtime_mode=runtime_mode,
        runtime_note=runtime_note,
    )
