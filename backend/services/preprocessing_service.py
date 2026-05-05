from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.utils.config import settings


@dataclass
class PreprocessResult:
    normalized_image: np.ndarray
    preview_rgb: np.ndarray
    original_shape: tuple[int, ...]
    file_format: str
    modality: str
    is_3d: bool
    voxel_spacing: tuple[float, float, float] | None
    volume_data: np.ndarray | None = None
    selected_slice_index: int | None = None
    is_area_based_approximation: bool = True
    modality_channels: int = 1
    preprocessing_notes: list[str] | None = None


SUPPORTED_2D = {".jpg", ".jpeg", ".png"}
SUPPORTED_DICOM = {".dcm"}


def _normalize_intensity(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.size == 0:
        return image

    lower = float(np.percentile(image, 1.0))
    upper = float(np.percentile(image, 99.0))
    if upper - lower < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    clipped = np.clip(image, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def _largest_component(binary_mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def _skull_strip_2d(image: np.ndarray) -> np.ndarray:
    if settings.skull_strip_mode.lower() == "none":
        return image

    img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _largest_component((thresh > 0).astype(np.uint8))
    stripped = (img_u8 * mask).astype(np.uint8)
    return _normalize_intensity(stripped)


def _slice_from_volume(volume: np.ndarray) -> tuple[np.ndarray, int]:
    z_idx = int(volume.shape[2] // 2)
    image = volume[:, :, z_idx]
    return image, z_idx


def _load_nifti_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None, int]:
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("nibabel is required to load NIfTI files.") from exc

    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3]) if nii.header else None

    channels = 1
    if data.ndim == 3:
        volume = data
    elif data.ndim == 4:
        channels = int(data.shape[3])
        idx = int(np.clip(settings.nifti_modality_index, 0, channels - 1))
        volume = data[:, :, :, idx]
    else:
        raise ValueError(f"Unsupported NIfTI dimensions: {data.shape}")

    return volume.astype(np.float32), spacing, channels


def _load_dicom(path: Path) -> np.ndarray:
    try:
        import pydicom
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pydicom is required to load DICOM files.") from exc

    dcm = pydicom.dcmread(str(path))
    image = dcm.pixel_array.astype(np.float32)
    if image.ndim > 2:
        image = np.squeeze(image)
    return image


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _resize_2d(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _prepare_2d_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    norm = _normalize_intensity(image)
    denoised = cv2.GaussianBlur(norm, (5, 5), 0)
    stripped = _skull_strip_2d(denoised)
    return _resize_2d(stripped, target_size)


def _prepare_3d_volume(volume: np.ndarray, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    normalized_volume = _normalize_intensity(volume)
    preview_slice, z_idx = _slice_from_volume(normalized_volume)
    prepared_slice = _prepare_2d_image(preview_slice, target_size)
    return normalized_volume, prepared_slice, z_idx


def _validate_mri_like_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Uploaded image is empty or unreadable.")

    if image.ndim == 3:
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)
        channel_delta = float((np.abs(b - g).mean() + np.abs(g - r).mean() + np.abs(b - r).mean()) / 3.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channel_delta > 18.0 and sat_mean > 35.0:
            raise ValueError(
                "Uploaded file appears to be a natural/color photo, not a grayscale brain MRI slice."
            )
    else:
        gray = image.astype(np.float32)

    gray = gray.astype(np.float32)
    g_min = float(np.min(gray))
    g_max = float(np.max(gray))
    if g_max - g_min > 1e-8:
        gray_u8 = np.clip((gray - g_min) * 255.0 / (g_max - g_min), 0, 255).astype(np.uint8)
    else:
        gray_u8 = np.zeros_like(gray, dtype=np.uint8)

    std = float(np.std(gray_u8))
    if std < 8.0:
        raise ValueError("Image contrast is too low for MRI analysis.")

    edges = cv2.Canny(gray_u8, 35, 120)
    edge_density = float(np.mean(edges > 0))
    if edge_density > 0.35:
        raise ValueError("Image texture is atypical for brain MRI. Please upload a valid MRI scan.")

    _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _largest_component((thresh > 0).astype(np.uint8))
    occupancy = float(mask.mean())
    if occupancy < 0.03 or occupancy > 0.95:
        raise ValueError("Image does not resemble a centered brain MRI slice.")


def preprocess_scan(image_path: str, target_size: tuple[int, int] = (224, 224)) -> PreprocessResult:
    path = Path(image_path)
    ext = path.suffix.lower()

    is_3d = False
    modality = "UNKNOWN"
    spacing = None
    volume_data = None
    selected_slice_index: int | None = None
    modality_channels = 1
    notes: list[str] = []

    if _is_nifti(path):
        volume, spacing, modality_channels = _load_nifti_volume(path)
        original_shape = tuple(volume.shape)
        volume_data, resized, selected_slice_index = _prepare_3d_volume(volume, target_size)
        is_3d = True
        modality = "MRI-3D-NIfTI"
        notes.append("3D NIfTI loaded and intensity-normalized with percentile clipping.")
        if modality_channels > 1:
            notes.append(
                f"Multi-modal NIfTI detected ({modality_channels} channels). "
                f"Using modality index {int(np.clip(settings.nifti_modality_index, 0, modality_channels - 1))}."
            )
    elif ext in SUPPORTED_DICOM:
        image = _load_dicom(path)
        _validate_mri_like_image(image)
        original_shape = tuple(image.shape)
        resized = _prepare_2d_image(image, target_size)
        modality = "DICOM-2D"
        notes.append("DICOM slice loaded and preprocessed in 2D mode.")
    elif ext in SUPPORTED_2D:
        image_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_color is None:
            raise ValueError("Unable to decode image file.")
        _validate_mri_like_image(image_color)
        image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        if image is None:
            raise ValueError("Unable to decode image file.")
        original_shape = tuple(image.shape)
        resized = _prepare_2d_image(image, target_size)
        modality = "MRI-2D"
        notes.append("2D image loaded and preprocessed.")
    else:
        raise ValueError(f"Unsupported file format: {path.name}")

    if resized.ndim != 2:
        raise ValueError("Preprocessed image is not 2D grayscale.")

    preview_rgb = cv2.cvtColor((np.clip(resized, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    is_area_based_approximation = not bool(is_3d and spacing and volume_data is not None)
    if is_area_based_approximation:
        notes.append("Tumor metric fallback: area-based approximation.")
    else:
        notes.append("3D tumor volume estimation enabled using voxel spacing metadata.")

    return PreprocessResult(
        normalized_image=resized.astype(np.float32),
        preview_rgb=preview_rgb,
        original_shape=original_shape,
        file_format=path.suffix.lower(),
        modality=modality,
        is_3d=is_3d,
        voxel_spacing=spacing,
        volume_data=volume_data,
        selected_slice_index=selected_slice_index,
        is_area_based_approximation=is_area_based_approximation,
        modality_channels=modality_channels,
        preprocessing_notes=notes,
    )
