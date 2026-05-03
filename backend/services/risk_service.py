from __future__ import annotations

from backend.utils.config import settings


def compute_risk_category(tumor_detected: bool, confidence: float, tumor_area: float) -> str:
    if not tumor_detected:
        return "Low"
    if confidence >= 0.85 or tumor_area >= settings.high_risk_area_threshold:
        return "High"
    if confidence >= 0.6 or tumor_area >= settings.medium_risk_area_threshold:
        return "Medium"
    return "Low"


def confidence_warning(confidence: float) -> str | None:
    if confidence < settings.low_confidence_threshold:
        return "Low confidence prediction. Expert review strongly recommended."
    return None
