from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    normalized_image: np.ndarray
    preview_rgb: np.ndarray
    original_shape: tuple[int, ...]
    file_format: str
    modality: str
    is_3d: bool
    voxel_spacing: tuple[float, float, float] | None


SUPPORTED_2D = {".jpg", ".jpeg", ".png"}
SUPPORTED_3D = {".nii", ".gz"}
SUPPORTED_DICOM = {".dcm"}


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_v = float(np.min(image))
    max_v = float(np.max(image))
    if max_v - min_v < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_v) / (max_v - min_v)


def _skull_strip_2d(image: np.ndarray) -> np.ndarray:
    img_u8 = (image * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_u8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    stripped = (img_u8 * (mask > 0)).astype(np.uint8)
    return _normalize_intensity(stripped)


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("nibabel is required to load NIfTI files.") from exc

    nii = nib.load(str(path))
    data = nii.get_fdata()
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3]) if nii.header else None
    if data.ndim == 3:
        z_idx = data.shape[2] // 2
        image = data[:, :, z_idx]
    elif data.ndim == 4:
        z_idx = data.shape[2] // 2
        image = data[:, :, z_idx, 0]
    else:
        raise ValueError("Unsupported NIfTI dimensionality.")
    return image.astype(np.float32), spacing


def _load_dicom(path: Path) -> np.ndarray:
    try:
        import pydicom
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pydicom is required to load DICOM files.") from exc

    dcm = pydicom.dcmread(str(path))
    image = dcm.pixel_array.astype(np.float32)
    return image


def preprocess_scan(image_path: str, target_size: tuple[int, int] = (224, 224)) -> PreprocessResult:
    path = Path(image_path)
    ext = path.suffix.lower()

    is_3d = False
    modality = "UNKNOWN"
    spacing = None

    if path.name.lower().endswith(".nii.gz") or ext in SUPPORTED_3D:
        raw = _load_nifti(path)
        if isinstance(raw, tuple):
            image, spacing = raw
        else:
            image = raw
        is_3d = True
        modality = "MRI-3D"
    elif ext in SUPPORTED_DICOM:
        image = _load_dicom(path)
        modality = "DICOM"
    elif ext in SUPPORTED_2D:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        modality = "MRI-2D"
        if image is None:
            raise ValueError("Unable to decode image file.")
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    original_shape = image.shape
    normalized = _normalize_intensity(image)
    denoised = cv2.GaussianBlur(normalized, (5, 5), 0)
    skull_stripped = _skull_strip_2d(denoised)
    resized = cv2.resize(skull_stripped, target_size, interpolation=cv2.INTER_AREA)
    preview_rgb = cv2.cvtColor((resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if resized.ndim != 2:
        raise ValueError("Preprocessed image is not 2D grayscale.")

    return PreprocessResult(
        normalized_image=resized,
        preview_rgb=preview_rgb,
        original_shape=tuple(original_shape),
        file_format=path.suffix.lower(),
        modality=modality,
        is_3d=is_3d,
        voxel_spacing=spacing,
    )
