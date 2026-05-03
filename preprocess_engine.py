"""Legacy compatibility wrapper.

Use backend.services.preprocessing_service.preprocess_scan for new code.
"""

from __future__ import annotations

import numpy as np

from backend.services.preprocessing_service import preprocess_scan


def apply_preprocessing(image_path: str, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    result = preprocess_scan(image_path=image_path, target_size=size)
    return result.normalized_image

