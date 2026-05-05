from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Patient, Scan, ScanProbability
from backend.services.rag_service import add_rag_document
from backend.services.storage_service import save_corrected_mask
from backend.schemas.patient import PatientRead
from backend.schemas.scan import ScanRead
from backend.utils.assets import to_storage_url, volume_manifest_to_urls


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
    volume_slice_urls, selected_slice_index = volume_manifest_to_urls(scan.volume_manifest_path)

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
            "corrected_mask_url": to_storage_url(scan.corrected_mask_path),
            "gradcam_url": to_storage_url(scan.gradcam_path),
            "overlay_url": to_storage_url(scan.overlay_path),
            "report_url": f"/api/reports/{scan.id}" if scan.report_path else None,
            "volume_manifest_url": to_storage_url(scan.volume_manifest_path),
        },
        "volume_slice_urls": volume_slice_urls,
        "selected_slice_index": selected_slice_index,
    }


@router.post("/api/scans/{scan_id}/correct-mask")
@router.post("/scans/{scan_id}/correct-mask")
async def correct_scan_mask(
    scan_id: int,
    corrected_mask: UploadFile = File(...),
    correction_notes: str | None = Form(None),
    corrected_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    scan = db.get(Scan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if not corrected_mask.filename:
        raise HTTPException(status_code=400, detail="Corrected mask file is required")

    ext = corrected_mask.filename.lower()
    if not (ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Corrected mask must be PNG/JPG/JPEG")

    content = await corrected_mask.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded corrected mask is empty")

    corrected_path = save_corrected_mask(content, scan_id=scan.id)
    scan.corrected_mask_path = corrected_path

    note_chunks = [chunk for chunk in [scan.correction_notes, correction_notes] if chunk]
    if corrected_by:
        note_chunks.append(f"Corrected by: {corrected_by} ({datetime.utcnow().isoformat()} UTC)")
    scan.correction_notes = "\n".join(note_chunks) if note_chunks else None

    db.commit()
    db.refresh(scan)

    patient = db.get(Patient, scan.patient_db_id)
    if patient:
        rag_text = (
            f"Mask correction logged for scan {scan.id} on {scan.scan_date}. "
            f"Corrected mask path stored. Notes: {scan.correction_notes or 'N/A'}"
        )
        add_rag_document(
            db,
            patient_db_id=patient.id,
            scan_id=scan.id,
            document_text=rag_text,
            document_type="mask_correction",
        )

    return {
        "scan_id": scan.id,
        "corrected_mask_path": scan.corrected_mask_path,
        "corrected_mask_url": to_storage_url(scan.corrected_mask_path),
        "correction_notes": scan.correction_notes,
        "message": "Corrected mask saved successfully for active-learning/review workflow.",
    }
