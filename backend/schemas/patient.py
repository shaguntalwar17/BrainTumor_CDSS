from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PatientCreate(BaseModel):
    patient_id: str
    patient_code: str | None = None
    name: str
    age: int
    gender: str
    contact: str | None = None


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: str
    patient_code: str | None
    name: str
    age: int
    gender: str
    contact: str | None
    created_at: datetime
