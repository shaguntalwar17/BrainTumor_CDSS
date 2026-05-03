from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Patient, Scan, ScanProbability
from backend.schemas.patient import PatientRead
from backend.schemas.scan import ScanRead
from backend.utils.assets import to_storage_url


router = APIRouter(tags=["scan"])


@router.get("/scans/{scan_id}")
@router.get("/api/scans/{scan_id}")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    patient = db.get(Patient, scan.patient_db_id)
    probabilities = db.scalars(
        select(ScanProbability).where(ScanProbability.scan_id == scan.id).order_by(ScanProbability.probability.desc())
    ).all()

    return {
        "scan": ScanRead.model_validate(scan),
        "patient": PatientRead.model_validate(patient) if patient else None,
        "runtime_mode": "trained" if "-trained" in scan.model_version.lower() else "demo",
        "class_probabilities": [
            {"class_name": row.class_name, "probability": row.probability}
            for row in probabilities
        ],
        "assets": {
            "image_url": to_storage_url(scan.image_path),
            "mask_url": to_storage_url(scan.mask_path),
            "gradcam_url": to_storage_url(scan.gradcam_path),
            "overlay_url": to_storage_url(scan.overlay_path),
            "report_url": f"/api/reports/{scan.id}" if scan.report_path else None,
        },
    }
