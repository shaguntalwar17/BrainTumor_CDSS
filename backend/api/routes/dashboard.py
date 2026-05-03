from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from backend.api.deps import get_db
from backend.models.entities import Patient, Scan


router = APIRouter(tags=["dashboard"])


@router.get("/dashboard/summary")
@router.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    total_patients = db.scalar(select(func.count(Patient.id))) or 0
    total_scans = db.scalar(select(func.count(Scan.id))) or 0
    tumor_detected_scans = db.scalar(select(func.count(Scan.id)).where(Scan.tumor_detected.is_(True))) or 0
    high_risk_scans = db.scalar(select(func.count(Scan.id)).where(Scan.risk_category == "High")) or 0

    latest_scans = db.scalars(select(Scan).order_by(Scan.created_at.desc()).limit(5)).all()
    recent = [
        {
            "scan_id": s.id,
            "patient_db_id": s.patient_db_id,
            "scan_date": s.scan_date,
            "tumor_detected": s.tumor_detected,
            "tumor_type": s.tumor_type,
            "risk_category": s.risk_category,
            "confidence_score": s.confidence_score,
        }
        for s in latest_scans
    ]

    return {
        "summary": {
            "total_patients": total_patients,
            "total_scans": total_scans,
            "tumor_detected_scans": tumor_detected_scans,
            "high_risk_scans": high_risk_scans,
        },
        "recent_scans": recent,
    }

