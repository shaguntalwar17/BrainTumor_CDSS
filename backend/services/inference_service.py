from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

from backend.services.preprocessing_service import PreprocessResult
from backend.utils.config import settings


TUMOR_CLASSES = ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]
MODEL_VERSION = "research-prototype-v3"
DEFAULT_STAGE_CLASSES = ["Low-grade", "High-grade"]


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
    stage_label: str | None
    stage_confidence: float | None
    stage_method: str
    explainability_consistency_score: float
    explainability_warning: str | None
    uncertainty_score: float | None
    uncertainty_std: float | None
    inference_time_sec: float
    model_version: str
    runtime_mode: str
    runtime_note: str
    xai_method: str
    is_area_based_approximation: bool
    volume_units: str


_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


def _checkpoint_path(path_like: str | None) -> Path | None:
    raw = (path_like or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def _candidate_classification_paths() -> list[Path]:
    paths: list[Path] = []
    primary = _checkpoint_path(settings.classification_model_path)
    if primary is not None:
        paths.append(primary)

    if settings.classification_ensemble_paths:
        for raw in settings.classification_ensemble_paths.split(","):
            candidate = _checkpoint_path(raw)
            if candidate is not None and candidate not in paths:
                paths.append(candidate)

    auto_candidates = [
        Path("ml/artifacts/classification_resnet50/best_classification_resnet50.pt"),
        Path("ml/artifacts/classification_efficientnet_b3/best_classification_efficientnet_b3.pt"),
        Path("ml/artifacts/classification_convnext_tiny/best_classification_convnext_tiny.pt"),
        Path("ml/artifacts/classification_vit_b16/best_classification_vit_b16.pt"),
    ]
    for candidate in auto_candidates:
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)
    return paths


def _norm_label(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _is_no_tumor_label(label: str) -> bool:
    key = _norm_label(label)
    return key in {"notumor", "normal", "healthy", "nontumor"}


def _find_no_tumor_key(probabilities: dict[str, float]) -> str | None:
    for key in probabilities:
        if _is_no_tumor_label(key):
            return key
    return None


def _parse_class_weight_map(raw: str, classes: list[str]) -> dict[str, float]:
    weights = {cls: 1.0 for cls in classes}
    if not raw.strip():
        return weights

    lookup = {_norm_label(cls): cls for cls in classes}
    for token in raw.split(","):
        chunk = token.strip()
        if not chunk:
            continue
        if "=" in chunk:
            left, right = chunk.split("=", 1)
        elif ":" in chunk:
            left, right = chunk.split(":", 1)
        else:
            continue

        label = left.strip()
        key = lookup.get(_norm_label(label))
        if not key:
            continue
        try:
            value = float(right.strip())
        except ValueError:
            continue
        weights[key] = max(0.01, value)
    return weights


def _apply_probability_weights(probabilities: np.ndarray, classes: list[str]) -> np.ndarray:
    weights = _parse_class_weight_map(settings.classification_prior_weights, classes)
    weight_vec = np.array([weights.get(cls, 1.0) for cls in classes], dtype=np.float32)
    calibrated = probabilities.astype(np.float32) * weight_vec
    total = float(calibrated.sum())
    if total <= 1e-8:
        return probabilities
    return calibrated / total


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def _segment_tumor_demo(image: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    threshold = int(np.clip(np.mean(image_u8) + 0.8 * np.std(image_u8), 25, 225))
    _, mask = cv2.threshold(image_u8, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _largest_component((mask > 0).astype(np.uint8))


def _segment_volume_demo(volume: np.ndarray) -> np.ndarray:
    h, w, d = volume.shape
    out = np.zeros((h, w, d), dtype=np.uint8)
    for i in range(d):
        out[:, :, i] = _segment_tumor_demo(volume[:, :, i])
    return out


def _detect(mask: np.ndarray, image: np.ndarray) -> tuple[bool, float]:
    area_ratio = float(mask.mean())
    texture = float(np.std(image))
    confidence = float(np.clip(0.2 + area_ratio * 12 + texture * 0.6, 0.05, 0.99))
    detected = area_ratio > 0.006 and confidence >= 0.45
    return detected, confidence


def _classify_demo(mask: np.ndarray, image: np.ndarray, detected: bool) -> tuple[str, dict[str, float], float]:
    if not detected:
        probs = {cls: 0.0 for cls in TUMOR_CLASSES}
        probs["No_Tumor"] = 1.0
        return "No_Tumor", probs, 0.01

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

    entropy = float(-(vec * np.log(vec + 1e-8)).sum() / np.log(len(vec)))
    return pred, probs, entropy


def _generate_xai_demo(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def _overlay(image: np.ndarray, mask: np.ndarray, xai_map: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heat_u8 = (np.clip(xai_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_rgb, 0.6, heat_color, 0.4, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def _explainability_consistency(mask: np.ndarray, xai_map: np.ndarray) -> tuple[float, str | None]:
    if not np.any(mask):
        return 1.0, None

    x = xai_map > np.percentile(xai_map, 70)
    m = mask > 0
    inter = float(np.logical_and(x, m).sum())
    union = float(np.logical_or(x, m).sum())
    score = inter / (union + 1e-8)

    if score < 0.25:
        warning = "Model attention may not be fully aligned with the segmented tumor region. Expert review is recommended."
    else:
        warning = None
    return float(score), warning


def _load_trained_runtime():
    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE:
            return _MODEL_CACHE

        try:
            import torch
            from ml.classification.train import build_model
            from ml.segmentation.models import build_segmentation_model
        except Exception as exc:
            raise ModelRuntimeError(
                "Required ML dependencies are missing for trained mode. Install dependencies or switch MODEL_RUNTIME_MODE=demo."
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ensemble_paths = _candidate_classification_paths()
        if not ensemble_paths:
            raise ModelRuntimeError(
                "No classification checkpoint found. Set CLASSIFICATION_MODEL_PATH or CLASSIFICATION_ENSEMBLE_PATHS."
            )

        classifiers: list[object] = []
        classes: list[str] | None = None
        cls_name_for_tag = "ensemble"
        for idx, path in enumerate(ensemble_paths):
            ckpt = torch.load(path, map_location=device)
            cls_name = str(ckpt.get("model_name", "resnext101_32x8d")).lower()
            ckpt_classes = list(ckpt.get("classes", TUMOR_CLASSES))
            if classes is None:
                classes = ckpt_classes
                cls_name_for_tag = cls_name
            elif ckpt_classes != classes:
                # Skip incompatible checkpoints to keep label mapping deterministic.
                continue
            classifier = build_model(cls_name, num_classes=len(ckpt_classes), dropout=settings.mc_dropout_rate)
            classifier.load_state_dict(ckpt["state_dict"], strict=False)
            classifier.to(device).eval()
            classifiers.append(classifier)

        if not classifiers or classes is None:
            raise ModelRuntimeError(
                "No compatible classification checkpoint could be loaded for trained inference."
            )
        if len(classifiers) > 1:
            cls_name_for_tag = "ensemble"

        segmenter = None
        seg_name = "demo_segmentation"
        seg_path = _checkpoint_path(settings.segmentation_model_path)
        if seg_path is not None:
            try:
                seg_ckpt = torch.load(seg_path, map_location=device)
                seg_name = str(seg_ckpt.get("model_name", "unet"))
                segmenter = build_segmentation_model(seg_name, in_channels=1, out_channels=1)
                segmenter.load_state_dict(seg_ckpt["state_dict"], strict=False)
                segmenter.to(device).eval()
            except Exception:
                segmenter = None
                seg_name = "demo_segmentation"

        stage_classifier = None
        stage_classes = [token.strip() for token in settings.stage_classes.split(",") if token.strip()]
        if not stage_classes:
            stage_classes = DEFAULT_STAGE_CLASSES
        stage_path = _checkpoint_path(settings.stage_model_path)
        if stage_path is not None:
            try:
                stage_ckpt = torch.load(stage_path, map_location=device)
                stage_name = str(stage_ckpt.get("model_name", settings.stage_model_name)).lower()
                stage_ckpt_classes = list(stage_ckpt.get("classes", stage_classes))
                stage_model = build_model(stage_name, num_classes=len(stage_ckpt_classes), dropout=0.2)
                stage_model.load_state_dict(stage_ckpt["state_dict"], strict=False)
                stage_model.to(device).eval()
                stage_classifier = stage_model
                stage_classes = stage_ckpt_classes
            except Exception:
                stage_classifier = None

        _MODEL_CACHE.update(
            {
                "device": device,
                "classifiers": classifiers,
                "classifier": classifiers[0],
                "segmenter": segmenter,
                "classes": classes,
                "classification_model_name": cls_name_for_tag,
                "segmentation_model_name": seg_name,
                "stage_classifier": stage_classifier,
                "stage_classes": stage_classes,
                "segmenter_available": segmenter is not None,
            }
        )
        return _MODEL_CACHE


def _to_classifier_tensor(image: np.ndarray, device):
    import torch

    image_rgb = cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)


def _enable_dropout_layers(model) -> None:
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _classify_trained_with_uncertainty(image: np.ndarray) -> tuple[str, dict[str, float], float, float]:
    runtime = _load_trained_runtime()
    import torch

    classifiers = runtime["classifiers"]
    device = runtime["device"]
    classes = runtime["classes"]
    temperature = max(1e-3, float(settings.classification_temperature))

    tensor = _to_classifier_tensor(image, device)
    n_samples = max(1, int(settings.mc_dropout_samples))
    use_tta = bool(settings.enable_tta_inference)

    for model in classifiers:
        model.eval()
        if n_samples > 1:
            _enable_dropout_layers(model)

    prob_samples: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            ensemble_probs = []
            for model in classifiers:
                view_tensors = [tensor]
                if use_tta:
                    view_tensors.append(torch.flip(tensor, dims=[3]))
                    view_tensors.append(torch.flip(tensor, dims=[2]))
                view_probs = []
                for view in view_tensors:
                    logits = model(view) / temperature
                    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    view_probs.append(probs)
                model_probs = np.mean(np.stack(view_probs, axis=0), axis=0)
                ensemble_probs.append(model_probs)
            prob_samples.append(np.mean(np.stack(ensemble_probs, axis=0), axis=0))

    for model in classifiers:
        model.eval()

    stacked = np.stack(prob_samples, axis=0)
    mean_probs_raw = stacked.mean(axis=0)
    var_probs = stacked.var(axis=0)
    mean_probs = _apply_probability_weights(mean_probs_raw, list(classes))

    pred_idx = int(np.argmax(mean_probs))
    pred_std = float(np.sqrt(max(float(var_probs[pred_idx]), 0.0)))
    entropy = float(-(mean_probs * np.log(mean_probs + 1e-8)).sum() / np.log(len(mean_probs)))

    named = {str(cls): float(prob) for cls, prob in zip(classes, mean_probs)}
    for c in TUMOR_CLASSES:
        named.setdefault(c, 0.0)

    pred = str(classes[pred_idx])
    return pred, named, entropy, pred_std


def _segment_trained(image: np.ndarray) -> np.ndarray:
    runtime = _load_trained_runtime()
    if runtime.get("segmenter") is None:
        return _segment_tumor_demo(image)

    import torch

    segmenter = runtime["segmenter"]
    device = runtime["device"]

    tensor = torch.from_numpy(image[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = segmenter(tensor)
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)
    return _largest_component(mask)


def _segment_volume_trained(volume: np.ndarray) -> np.ndarray:
    runtime = _load_trained_runtime()
    if runtime.get("segmenter") is None:
        return _segment_volume_demo(volume)

    import torch

    segmenter = runtime["segmenter"]
    device = runtime["device"]

    h, w, d = volume.shape
    out = np.zeros((h, w, d), dtype=np.uint8)
    with torch.no_grad():
        for idx in range(d):
            slice_img = volume[:, :, idx]
            resized = cv2.resize(slice_img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
            tensor = torch.from_numpy(resized[None, None, :, :]).to(device)
            logits = segmenter(tensor)
            prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            mask_small = (prob > 0.5).astype(np.uint8)
            mask_orig = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            out[:, :, idx] = _largest_component(mask_orig)
    return out


def _find_last_conv_layer(model):
    import torch.nn as nn

    for layer in reversed(list(model.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    raise RuntimeError("No Conv2d layer found for explainability generation.")


def _generate_trained_xai_map(image: np.ndarray, class_index: int) -> np.ndarray:
    runtime = _load_trained_runtime()
    import torch

    classifier = runtime["classifier"]
    device = runtime["device"]

    classifier.eval()
    tensor = _to_classifier_tensor(image, device)

    target_layer = _find_last_conv_layer(classifier)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _fwd_hook(_, __, output):
        activations["value"] = output

    def _bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        logits = classifier(tensor)
        score = logits[:, class_index].sum()
        classifier.zero_grad(set_to_none=True)
        score.backward()

        acts = activations["value"][0]
        grads = gradients["value"][0]

        method = settings.xai_method.strip().lower()
        if method == "layercam":
            cam = torch.relu(grads) * acts
            cam = cam.sum(dim=0)
        else:
            cam = torch.relu(grads * acts).sum(dim=0)

        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
    finally:
        h1.remove()
        h2.remove()

    cam_np = cv2.resize(cam_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np.astype(np.float32)


def _compute_tumor_volume(mask_volume: np.ndarray, voxel_spacing: tuple[float, float, float] | None) -> float | None:
    if voxel_spacing is None:
        return None
    voxel_volume_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    voxel_count = int(np.sum(mask_volume > 0))
    return float(voxel_count * voxel_volume_mm3)


def _proxy_stage_from_metric(
    tumor_detected: bool,
    tumor_type: str,
    tumor_area: float,
    tumor_volume: float | None,
) -> tuple[str | None, float | None, str]:
    if not tumor_detected:
        return "No Tumor", 1.0, "no_tumor"

    # Glioma-focused proxy with transparent limitations.
    metric = tumor_volume if tumor_volume is not None else tumor_area
    if tumor_type.lower() == "glioma":
        if metric < 2500:
            return "Stage I/II (Proxy)", 0.62, "proxy_volume_rule"
        if metric < 9000:
            return "Stage II/III (Proxy)", 0.68, "proxy_volume_rule"
        return "Stage III/IV (Proxy)", 0.74, "proxy_volume_rule"

    if metric < 1800:
        return "Early Burden Pattern (Proxy)", 0.56, "proxy_burden_rule"
    if metric < 7000:
        return "Intermediate Burden Pattern (Proxy)", 0.6, "proxy_burden_rule"
    return "Advanced Burden Pattern (Proxy)", 0.64, "proxy_burden_rule"


def _estimate_stage_with_model(image: np.ndarray) -> tuple[str | None, float | None, str]:
    runtime = _load_trained_runtime()
    stage_classifier = runtime.get("stage_classifier")
    if stage_classifier is None:
        return None, None, "stage_model_unavailable"

    import torch

    device = runtime["device"]
    stage_classes: Iterable[str] = runtime.get("stage_classes") or DEFAULT_STAGE_CLASSES
    stage_class_list = list(stage_classes)
    tensor = _to_classifier_tensor(image, device)
    with torch.no_grad():
        logits = stage_classifier(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    idx = int(np.argmax(probs))
    label = stage_class_list[idx] if idx < len(stage_class_list) else f"StageClass_{idx}"
    return str(label), float(probs[idx]), "stage_model"


def _resolve_runtime_mode() -> str:
    configured = settings.model_runtime_mode.strip().lower()
    if configured in {"demo", "trained"}:
        return configured

    # Auto mode: use trained runtime only when a trained classification checkpoint is available.
    return "trained" if _candidate_classification_paths() else "demo"


def _apply_pituitary_guard(
    predicted_label: str,
    class_probabilities: dict[str, float],
    mask_slice: np.ndarray,
) -> tuple[str, str | None]:
    if not settings.pituitary_guard_enabled:
        return predicted_label, None

    if _norm_label(predicted_label) != _norm_label("Pituitary"):
        return predicted_label, None

    no_tumor_key = _find_no_tumor_key(class_probabilities)
    tumor_pairs = sorted(
        (
            (label, float(prob))
            for label, prob in class_probabilities.items()
            if label != no_tumor_key
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(tumor_pairs) < 2:
        return predicted_label, None

    top_label, top_prob = tumor_pairs[0]
    second_label, second_prob = tumor_pairs[1]
    no_tumor_prob = float(class_probabilities.get(no_tumor_key, 0.0)) if no_tumor_key else 0.0
    if _norm_label(top_label) != _norm_label("Pituitary"):
        return predicted_label, None

    margin = float(top_prob - second_prob)

    if mask_slice.size == 0 or not np.any(mask_slice):
        mismatch = True
    else:
        ys, xs = np.where(mask_slice > 0)
        x_norm = float(np.mean(xs) / max(mask_slice.shape[1] - 1, 1))
        y_norm = float(np.mean(ys) / max(mask_slice.shape[0] - 1, 1))
        area_ratio = float(np.mean(mask_slice > 0))
        mismatch = not (
            settings.pituitary_centroid_x_min <= x_norm <= settings.pituitary_centroid_x_max
            and settings.pituitary_centroid_y_min <= y_norm <= settings.pituitary_centroid_y_max
            and area_ratio <= settings.pituitary_area_ratio_max
        )

    if not mismatch and margin > float(settings.pituitary_guard_margin):
        return predicted_label, None

    weak_confidence = top_prob < 0.58 and no_tumor_prob >= 0.25
    margin_small = margin <= float(settings.pituitary_guard_margin)
    if not mismatch and not weak_confidence:
        return predicted_label, None
    if not (margin_small or weak_confidence):
        return predicted_label, None

    note = (
        "Pituitary guard adjusted classification due to weak margin and atypical segmented location/size."
    )
    return second_label, note


def _fuse_detection_with_classification(
    mask_slice: np.ndarray,
    image: np.ndarray,
    tumor_type: str,
    class_probabilities: dict[str, float],
) -> tuple[bool, float]:
    mask_detected, mask_conf = _detect(mask_slice, image)
    no_tumor_key = _find_no_tumor_key(class_probabilities)
    no_tumor_prob = float(class_probabilities.get(no_tumor_key, 0.0)) if no_tumor_key else 0.0
    tumor_probs = [
        float(prob)
        for cls, prob in class_probabilities.items()
        if cls != no_tumor_key
    ]
    tumor_prob = max(tumor_probs) if tumor_probs else 0.0

    area_pixels = int(np.sum(mask_slice > 0))
    area_ratio = float(area_pixels / max(mask_slice.size, 1))
    morphology_positive = area_pixels >= int(settings.min_tumor_pixels) and area_ratio >= float(settings.min_tumor_area_ratio)
    classification_positive = (
        tumor_prob >= float(settings.tumor_detection_threshold)
        and no_tumor_prob <= float(settings.no_tumor_probability_threshold)
    )

    if _is_no_tumor_label(tumor_type):
        detected = bool(classification_positive and morphology_positive and tumor_prob > no_tumor_prob)
        if not detected and morphology_positive:
            recovered_threshold = max(float(settings.tumor_detection_threshold) * 0.75, no_tumor_prob * 0.55)
            detected = bool(tumor_prob >= recovered_threshold)
    else:
        detected = bool((classification_positive and (morphology_positive or tumor_prob >= 0.75)) or (mask_detected and tumor_prob >= 0.4))

    confidence = max(mask_conf, tumor_prob if detected else no_tumor_prob)
    return detected, float(np.clip(confidence, 0.0, 1.0))


def analyze_scan(preprocessed: PreprocessResult) -> AnalysisOutput:
    start = perf_counter()

    image = preprocessed.normalized_image
    runtime_mode = _resolve_runtime_mode()
    uncertainty_score = None
    uncertainty_std = None
    guard_note: str | None = None

    if runtime_mode == "trained":
        try:
            runtime = _load_trained_runtime()
        except ModelRuntimeError:
            # In auto/trained fallback scenarios we prefer graceful degradation to demo mode.
            runtime = {}
            runtime_mode = "demo"

    if runtime_mode == "trained":
        if preprocessed.volume_data is not None and preprocessed.volume_data.ndim == 3:
            mask_volume = _segment_volume_trained(preprocessed.volume_data)
            z_idx = preprocessed.selected_slice_index or (mask_volume.shape[2] // 2)
            mask_slice = cv2.resize(mask_volume[:, :, z_idx], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_volume = None
            mask_slice = _segment_trained(image)

        tumor_type, probs, uncertainty_score, uncertainty_std = _classify_trained_with_uncertainty(image)
        tumor_type, guard_note = _apply_pituitary_guard(tumor_type, probs, mask_slice)
        tumor_detected, confidence = _fuse_detection_with_classification(mask_slice, image, tumor_type, probs)
        if tumor_detected and _is_no_tumor_label(tumor_type):
            no_tumor_key = _find_no_tumor_key(probs)
            tumor_candidates = [
                (label, float(prob))
                for label, prob in probs.items()
                if label != no_tumor_key
            ]
            if tumor_candidates:
                tumor_type = max(tumor_candidates, key=lambda item: item[1])[0]
        no_tumor_key = _find_no_tumor_key(probs) or "No_Tumor"
        if not tumor_detected:
            tumor_type = no_tumor_key
            mask_slice = np.zeros_like(mask_slice, dtype=np.uint8)
            probs[no_tumor_key] = max(float(probs.get(no_tumor_key, 0.0)), 0.62)

        classes = list(runtime.get("classes", []))
        if tumor_type in classes:
            class_idx = classes.index(tumor_type)
        else:
            class_idx = int(np.argmax([float(probs.get(cls, 0.0)) for cls in classes])) if classes else 0
        try:
            xai_map = _generate_trained_xai_map(image, class_idx)
        except Exception:
            xai_map = _generate_xai_demo(image, mask_slice)

        model_version = f"{MODEL_VERSION}-trained"
        cls_mode = str(runtime.get("classification_model_name", "unknown"))
        seg_mode = "trained" if runtime.get("segmenter_available") else "demo_fallback"
        stage_mode = "trained" if runtime.get("stage_classifier") is not None else "proxy_fallback"
        runtime_note = (
            f"Trained classification ({cls_mode}) active with {seg_mode} segmentation and {stage_mode} stage estimation. "
            "Uncertainty uses Monte Carlo dropout; outputs remain research-only and require expert verification."
        )
        if guard_note:
            runtime_note = f"{runtime_note} {guard_note}"
    else:
        if preprocessed.volume_data is not None and preprocessed.volume_data.ndim == 3:
            mask_volume = _segment_volume_demo(preprocessed.volume_data)
            z_idx = preprocessed.selected_slice_index or (mask_volume.shape[2] // 2)
            mask_slice = cv2.resize(mask_volume[:, :, z_idx], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_volume = None
            mask_slice = _segment_tumor_demo(image)

        mask_detected, confidence = _detect(mask_slice, image)
        tumor_type, probs, uncertainty_score = _classify_demo(mask_slice, image, mask_detected)
        tumor_type, guard_note = _apply_pituitary_guard(tumor_type, probs, mask_slice)
        tumor_detected, confidence = _fuse_detection_with_classification(mask_slice, image, tumor_type, probs)
        if tumor_detected and _is_no_tumor_label(tumor_type):
            no_tumor_key = _find_no_tumor_key(probs)
            tumor_candidates = [
                (label, float(prob))
                for label, prob in probs.items()
                if label != no_tumor_key
            ]
            if tumor_candidates:
                tumor_type = max(tumor_candidates, key=lambda item: item[1])[0]
        no_tumor_key = _find_no_tumor_key(probs) or "No_Tumor"
        if not tumor_detected:
            mask_slice = np.zeros_like(mask_slice, dtype=np.uint8)
            tumor_type = no_tumor_key
            probs[no_tumor_key] = max(float(probs.get(no_tumor_key, 0.0)), 0.62)
        uncertainty_std = None
        xai_map = _generate_xai_demo(image, mask_slice)
        runtime_note = (
            "Demo Mode: heuristic inference and explainability pipeline. Stage output uses transparent proxy burden rules. "
            "Not clinically validated."
        )
        if guard_note:
            runtime_note = f"{runtime_note} {guard_note}"
        model_version = f"{MODEL_VERSION}-demo"
        runtime_mode = "demo"

    tumor_area = float(np.sum(mask_slice > 0))

    tumor_volume = None
    if mask_volume is not None:
        tumor_volume = _compute_tumor_volume(mask_volume, preprocessed.voxel_spacing)
    elif preprocessed.is_3d and preprocessed.voxel_spacing:
        sx, sy, sz = preprocessed.voxel_spacing
        tumor_volume = float(tumor_area * sx * sy * sz)

    stage_label = None
    stage_confidence = None
    stage_method = "stage_unavailable"
    if runtime_mode == "trained":
        stage_label, stage_confidence, stage_method = _estimate_stage_with_model(image)
    if not stage_label and settings.stage_proxy_enabled:
        stage_label, stage_confidence, stage_method = _proxy_stage_from_metric(
            tumor_detected=tumor_detected,
            tumor_type=tumor_type,
            tumor_area=tumor_area,
            tumor_volume=tumor_volume,
        )

    overlay = _overlay(image, mask_slice, xai_map)
    xai_score, xai_warning = _explainability_consistency(mask_slice, xai_map)
    infer_time = perf_counter() - start

    no_tumor_key_final = _find_no_tumor_key(probs) or "No_Tumor"
    if _is_no_tumor_label(tumor_type):
        confidence = max(confidence, probs.get(no_tumor_key_final, 0.0))
    else:
        confidence = max(confidence, probs.get(tumor_type, 0.0))

    return AnalysisOutput(
        tumor_detected=tumor_detected,
        tumor_type=tumor_type,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        class_probabilities=probs,
        mask=mask_slice.astype(np.uint8),
        gradcam=xai_map,
        overlay=overlay,
        tumor_area=tumor_area,
        tumor_volume=tumor_volume,
        stage_label=stage_label,
        stage_confidence=stage_confidence,
        stage_method=stage_method,
        explainability_consistency_score=xai_score,
        explainability_warning=xai_warning,
        uncertainty_score=uncertainty_score,
        uncertainty_std=uncertainty_std,
        inference_time_sec=float(infer_time),
        model_version=model_version,
        runtime_mode=runtime_mode,
        runtime_note=runtime_note,
        xai_method=settings.xai_method.strip().lower(),
        is_area_based_approximation=preprocessed.is_area_based_approximation,
        volume_units="mm^3" if tumor_volume is not None else "pixels",
    )
