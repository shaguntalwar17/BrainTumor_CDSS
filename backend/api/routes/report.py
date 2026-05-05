from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Patient, Scan
from backend.services.comparison_service import build_growth_change_map, build_growth_chart, compare_scans
from backend.services.report_service import generate_comparison_report
from backend.utils.assets import to_storage_url


router = APIRouter(tags=["report"])


@router.get("/api/reports")
@router.get("/reports")
def list_reports(
    patient_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    patient_lookup = None
    if patient_id:
        patient_lookup = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
        if not patient_lookup:
            return {"items": []}

    query = select(Scan).order_by(Scan.created_at.desc())
    if patient_lookup:
        query = query.where(Scan.patient_db_id == patient_lookup.id)
    scans = db.scalars(query).all()

    items = []
    for scan in scans:
        if not scan.report_path:
            continue
        patient = db.get(Patient, scan.patient_db_id)
        items.append(
            {
                "scan_id": scan.id,
                "patient_id": patient.patient_id if patient else None,
                "patient_name": patient.name if patient else None,
                "scan_date": scan.scan_date,
                "tumor_detected": scan.tumor_detected,
                "tumor_type": scan.tumor_type,
                "stage_label": scan.stage_label,
                "risk_category": scan.risk_category,
                "report_url": f"/api/reports/{scan.id}",
                "report_storage_url": to_storage_url(scan.report_path),
                "created_at": scan.created_at,
            }
        )
    return {"items": items}


@router.get("/report/{scan_id}")
@router.get("/api/reports/{scan_id}")
def download_report(scan_id: int, db: Session = Depends(get_db)):
    scan = db.get(Scan, scan_id)
    if not scan or not scan.report_path:
        raise HTTPException(status_code=404, detail="Report not found")

    report_path = Path(scan.report_path)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file missing")

    return FileResponse(str(report_path), media_type="application/pdf", filename=report_path.name)


@router.get("/api/reports/comparison/{patient_id}/{previous_scan_id}/{current_scan_id}")
@router.get("/reports/comparison/{patient_id}/{previous_scan_id}/{current_scan_id}")
def download_comparison_report(
    patient_id: str,
    previous_scan_id: int,
    current_scan_id: int,
    db: Session = Depends(get_db),
):
    patient = db.scalar(select(Patient).where(Patient.patient_id == patient_id))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    previous = db.get(Scan, previous_scan_id)
    current = db.get(Scan, current_scan_id)
    if not previous or not current:
        raise HTTPException(status_code=404, detail="Scan(s) not found")
    if previous.patient_db_id != patient.id or current.patient_db_id != patient.id:
        raise HTTPException(status_code=400, detail="Both scans must belong to the specified patient")

    cmp = compare_scans(previous, current)
    all_scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id)).all()
    chart_path = build_growth_chart(patient_id=patient.patient_id, scans=all_scans)
    growth_map_path = build_growth_change_map(
        patient_id=patient.patient_id,
        previous_scan_id=previous.id,
        current_scan_id=current.id,
        previous_mask_path=previous.mask_path,
        current_mask_path=current.mask_path,
    )
    comparison_report_path = generate_comparison_report(
        patient=patient,
        previous_scan=previous,
        current_scan=current,
        comparison_summary=cmp.summary,
        progression_status=cmp.progression_status,
        absolute_change=cmp.absolute_change,
        percentage_change=cmp.percentage_change,
        progression_chart_path=chart_path,
        growth_map_path=growth_map_path,
    )

    report_path = Path(comparison_report_path)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Comparison report file missing")
    return FileResponse(
        str(report_path),
        media_type="application/pdf",
        filename=report_path.name,
    )
