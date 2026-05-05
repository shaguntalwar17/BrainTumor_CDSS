from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from backend.utils.config import settings
from backend.utils.pathing import ensure_dir


for _path in [
    settings.storage_root,
    settings.upload_dir,
    settings.mask_dir,
    settings.gradcam_dir,
    settings.overlay_dir,
    settings.volume_preview_dir,
    settings.report_dir,
    settings.chart_dir,
]:
    ensure_dir(_path)


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


async def save_upload_file(file: UploadFile, patient_id: str, scan_date: date) -> str:
    filename_in = (file.filename or "").strip()
    if not filename_in:
        raise ValueError("Uploaded file must include a valid filename.")

    original_name = filename_in.lower()
    allowed_ext = {".png", ".jpg", ".jpeg", ".nii", ".nii.gz", ".dcm"}
    if original_name.endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = Path(filename_in).suffix.lower()

    if ext not in allowed_ext:
        raise ValueError("Unsupported file type. Please upload PNG, JPG, NIfTI (.nii/.nii.gz), or DICOM (.dcm).")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_stem(patient_id)}_{scan_date.isoformat()}_{timestamp}{ext}"
    out_path = Path(settings.upload_dir) / filename
    content = await file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")
    out_path.write_bytes(content)
    return out_path.as_posix()


def save_mask(mask: np.ndarray, scan_id: int) -> str:
    out_path = Path(settings.mask_dir) / f"scan_{scan_id}_mask.png"
    cv2.imwrite(str(out_path), (mask > 0).astype(np.uint8) * 255)
    return out_path.as_posix()


def save_gradcam(gradcam: np.ndarray, scan_id: int) -> str:
    out_path = Path(settings.gradcam_dir) / f"scan_{scan_id}_gradcam.png"
    grad_u8 = np.clip(gradcam * 255, 0, 255).astype(np.uint8)
    grad_col = cv2.applyColorMap(grad_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), grad_col)
    return out_path.as_posix()


def save_overlay(overlay: np.ndarray, scan_id: int) -> str:
    out_path = Path(settings.overlay_dir) / f"scan_{scan_id}_overlay.png"
    cv2.imwrite(str(out_path), overlay)
    return out_path.as_posix()


def save_corrected_mask(mask_bytes: bytes, scan_id: int) -> str:
    out_path = Path(settings.mask_dir) / f"scan_{scan_id}_corrected_mask.png"
    out_path.write_bytes(mask_bytes)
    return out_path.as_posix()


def save_volume_slice_stack(
    volume: np.ndarray,
    scan_id: int,
    selected_slice_index: int | None = None,
    max_slices: int = 48,
) -> tuple[str, list[str], int]:
    if volume.ndim != 3:
        raise ValueError("Volume stack export requires a 3D array.")

    h, w, depth = volume.shape
    out_dir = Path(settings.volume_preview_dir) / f"scan_{scan_id}"
    ensure_dir(out_dir)

    slice_indices: list[int]
    if depth <= max_slices:
        slice_indices = list(range(depth))
    else:
        slice_indices = np.linspace(0, depth - 1, num=max_slices, dtype=int).tolist()

    rel_paths: list[str] = []
    for z in slice_indices:
        sl = volume[:, :, z]
        sl_norm = sl.astype(np.float32)
        min_v = float(np.min(sl_norm))
        max_v = float(np.max(sl_norm))
        if max_v - min_v > 1e-8:
            sl_norm = (sl_norm - min_v) / (max_v - min_v)
        else:
            sl_norm = np.zeros_like(sl_norm, dtype=np.float32)
        sl_u8 = (np.clip(sl_norm, 0.0, 1.0) * 255).astype(np.uint8)
        out_path = out_dir / f"slice_{z:04d}.png"
        cv2.imwrite(str(out_path), sl_u8)
        rel_paths.append(out_path.as_posix())

    chosen = selected_slice_index if selected_slice_index is not None else depth // 2
    manifest = {
        "scan_id": scan_id,
        "volume_shape": [int(h), int(w), int(depth)],
        "selected_slice_index": int(chosen),
        "slice_indices": [int(v) for v in slice_indices],
        "slice_paths": rel_paths,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path.as_posix(), rel_paths, int(chosen)
