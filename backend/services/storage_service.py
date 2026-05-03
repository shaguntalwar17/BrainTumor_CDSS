from __future__ import annotations

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
    settings.report_dir,
    settings.chart_dir,
]:
    ensure_dir(_path)


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


async def save_upload_file(file: UploadFile, patient_id: str, scan_date: date) -> str:
    ext = Path(file.filename or "scan.bin").suffix or ".bin"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{_safe_stem(patient_id)}_{scan_date.isoformat()}_{timestamp}{ext}"
    out_path = Path(settings.upload_dir) / filename
    content = await file.read()
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
