from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class ScanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_db_id: int
    scan_date: date
    image_path: str
    mask_path: str | None
    gradcam_path: str | None
    report_path: str | None
    overlay_path: str | None
    volume_manifest_path: str | None
    corrected_mask_path: str | None
    tumor_detected: bool
    tumor_type: str | None
    confidence_score: float
    uncertainty_score: float | None
    uncertainty_std: float | None
    tumor_area: float
    tumor_volume: float | None
    stage_label: str | None
    stage_confidence: float | None
    stage_method: str | None
    risk_category: str
    explainability_consistency_score: float | None
    xai_method: str | None
    model_version: str
    radiologist_notes: str | None
    correction_notes: str | None
    created_at: datetime


class UploadScanResponse(BaseModel):
    patient_id: str
    patient_name: str
    generated_patient_id: bool = False
    matched_existing_patient: bool = False
    patient_match_strategy: str | None = None
    patient_match_score: float | None = None
    scan_id: int
    scan_date: date
    tumor_detected: bool
    tumor_type: str | None
    confidence_score: float
    uncertainty_score: float | None = None
    uncertainty_std: float | None = None
    tumor_area: float
    tumor_volume: float | None
    stage_label: str | None = None
    stage_confidence: float | None = None
    stage_method: str | None = None
    volume_units: str | None = None
    is_area_based_approximation: bool = True
    risk_category: str
    uncertainty_warning: str | None
    progression_status: str
    explainability_consistency_score: float | None
    explainability_warning: str | None = None
    xai_method: str | None = None
    longitudinal_tumor_progression_index: float | None
    model_version: str
    runtime_mode: str = "demo"
    runtime_note: str | None = None
    class_probabilities: list[ClassProbability]
    report_path: str
    gradcam_path: str | None
    mask_path: str | None
    overlay_path: str | None
    image_url: str | None = None
    report_url: str | None = None
    gradcam_url: str | None = None
    mask_url: str | None = None
    overlay_url: str | None = None
    volume_manifest_url: str | None = None
    volume_slice_urls: list[str] = Field(default_factory=list)
    selected_slice_index: int | None = None
    disclaimer: str
    stage_note: str
    attribution: str


class CompareScansRequest(BaseModel):
    patient_id: str
    previous_scan_id: int
    current_scan_id: int


class CompareScansResponse(BaseModel):
    patient_id: str
    previous_scan_id: int
    current_scan_id: int
    previous_scan_date: date
    current_scan_date: date
    previous_tumor_area: float
    current_tumor_area: float
    previous_tumor_volume: float | None
    current_tumor_volume: float | None
    absolute_change: float
    percentage_change: float
    tumor_type_change: str
    confidence_difference: float
    previous_stage_label: str | None = None
    current_stage_label: str | None = None
    stage_change: str | None = None
    previous_risk_level: str | None = None
    current_risk_level: str | None = None
    risk_level_change: str | None = None
    progression_status: str
    longitudinal_tumor_progression_index: float | None
    summary: str
    previous_scan_assets: dict[str, str | None] | None = None
    current_scan_assets: dict[str, str | None] | None = None
    progression_chart_url: str | None = None
    growth_map_url: str | None = None
