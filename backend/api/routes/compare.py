from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Comparison, Patient, Scan
from backend.schemas.scan import CompareScansRequest, CompareScansResponse
from backend.services.comparison_service import build_growth_chart, compare_scans
from backend.utils.assets import to_storage_url


router = APIRouter(tags=["comparison"])


def _scan_assets(scan: Scan) -> dict[str, str | None]:
    return {
        "image_url": to_storage_url(scan.image_path),
        "mask_url": to_storage_url(scan.mask_path),
        "gradcam_url": to_storage_url(scan.gradcam_path),
        "overlay_url": to_storage_url(scan.overlay_path),
        "report_url": f"/api/reports/{scan.id}" if scan.report_path else None,
    }


@router.post("/compare-scans", response_model=CompareScansResponse)
@router.post("/api/scans/compare", response_model=CompareScansResponse)
def compare_scans_route(payload: CompareScansRequest, db: Session = Depends(get_db)):
    patient = db.scalar(select(Patient).where(Patient.patient_id == payload.patient_id))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    previous = db.get(Scan, payload.previous_scan_id)
    current = db.get(Scan, payload.current_scan_id)

    if not previous or not current:
        raise HTTPException(status_code=404, detail="Scan(s) not found")
    if previous.patient_db_id != patient.id or current.patient_db_id != patient.id:
        raise HTTPException(status_code=400, detail="Scan IDs do not belong to requested patient")

    result = compare_scans(previous, current)

    existing = db.scalar(
        select(Comparison).where(
            Comparison.previous_scan_id == previous.id,
            Comparison.current_scan_id == current.id,
        )
    )
    if not existing:
        db.add(
            Comparison(
                patient_db_id=patient.id,
                previous_scan_id=result.previous_scan_id,
                current_scan_id=result.current_scan_id,
                previous_volume=result.previous_tumor_volume,
                current_volume=result.current_tumor_volume,
                absolute_change=result.absolute_change,
                percentage_change=result.percentage_change,
                progression_status=result.progression_status,
                longitudinal_index=result.longitudinal_tumor_progression_index,
                summary=result.summary,
            )
        )
        db.commit()

    all_scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id)).all()
    chart_path = build_growth_chart(patient_id=patient.patient_id, scans=all_scans)

    return CompareScansResponse(
        patient_id=payload.patient_id,
        previous_scan_id=result.previous_scan_id,
        current_scan_id=result.current_scan_id,
        previous_scan_date=result.previous_scan_date,
        current_scan_date=result.current_scan_date,
        previous_tumor_area=result.previous_tumor_area,
        current_tumor_area=result.current_tumor_area,
        previous_tumor_volume=result.previous_tumor_volume,
        current_tumor_volume=result.current_tumor_volume,
        absolute_change=result.absolute_change,
        percentage_change=result.percentage_change,
        tumor_type_change=result.tumor_type_change,
        confidence_difference=result.confidence_difference,
        previous_risk_level=result.previous_risk_level,
        current_risk_level=result.current_risk_level,
        risk_level_change=result.risk_level_change,
        progression_status=result.progression_status,
        longitudinal_tumor_progression_index=result.longitudinal_tumor_progression_index,
        summary=result.summary,
        previous_scan_assets=_scan_assets(previous),
        current_scan_assets=_scan_assets(current),
        progression_chart_url=to_storage_url(chart_path),
    )
