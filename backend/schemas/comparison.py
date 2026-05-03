from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ComparisonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_db_id: int
    previous_scan_id: int
    current_scan_id: int
    previous_volume: float | None
    current_volume: float | None
    absolute_change: float
    percentage_change: float
    progression_status: str
    longitudinal_index: float | None
    summary: str
    created_at: datetime
