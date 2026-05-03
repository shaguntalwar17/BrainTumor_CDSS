from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Patient, Scan, ScanProbability
from backend.schemas.patient import PatientRead
from backend.schemas.scan import ScanRead
from backend.utils.assets import to_storage_url


router = APIRouter(tags=["patients"])


def _scan_payload(db: Session, scan: Scan) -> dict:
    probs = db.scalars(
        select(ScanProbability).where(ScanProbability.scan_id == scan.id).order_by(ScanProbability.probability.desc())
    ).all()
    return {
        "scan": ScanRead.model_validate(scan),
        "class_probabilities": [{"class_name": p.class_name, "probability": p.probability} for p in probs],
        "assets": {
            "image_url": to_storage_url(scan.image_path),
            "mask_url": to_storage_url(scan.mask_path),
            "gradcam_url": to_storage_url(scan.gradcam_path),
            "overlay_url": to_storage_url(scan.overlay_path),
            "report_url": f"/api/reports/{scan.id}" if scan.report_path else None,
        },
    }


@router.get("/patients", response_model=list[PatientRead])
@router.get("/api/patients", response_model=list[PatientRead])
def list_patients(db: Session = Depends(get_db)):
    return db.scalars(select(Patient).order_by(Patient.created_at.desc())).all()


@router.get("/patients/{patient_id}")
@router.get("/api/patients/{patient_id}")
def patient_profile(patient_id: str, db: Session = Depends(get_db)):
    patient = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id).order_by(Scan.scan_date.asc())).all()

    return {
        "patient": PatientRead.model_validate(patient),
        "scans": [_scan_payload(db, s) for s in scans],
    }


@router.get("/patients/{patient_id}/scans")
@router.get("/api/patients/{patient_id}/scans")
def patient_scans(patient_id: str, db: Session = Depends(get_db)):
    patient = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id).order_by(Scan.scan_date.asc())).all()
    return [_scan_payload(db, s) for s in scans]

