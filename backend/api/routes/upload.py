from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Comparison, Patient, Scan, ScanProbability
from backend.schemas.scan import ClassProbability, UploadScanResponse
from backend.services.comparison_service import build_growth_chart, compare_scans
from backend.services.inference_service import ModelRuntimeError, analyze_scan
from backend.services.patient_service import resolve_or_create_patient
from backend.services.preprocessing_service import preprocess_scan
from backend.services.rag_service import add_rag_document
from backend.services.report_service import generate_report
from backend.services.risk_service import compute_risk_category, confidence_warning
from backend.services.storage_service import (
    save_gradcam,
    save_mask,
    save_overlay,
    save_upload_file,
    save_volume_slice_stack,
)
from backend.utils.assets import to_storage_url, volume_manifest_to_urls
from backend.utils.disclaimer import ATTRIBUTION, STAGE_NOTE, UI_DISCLAIMER


router = APIRouter(tags=["scan"])


@router.post("/upload-scan", response_model=UploadScanResponse)
@router.post("/api/scans/upload", response_model=UploadScanResponse)
async def upload_scan(
    patient_id: str | None = Form(None),
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    scan_date: str = Form(...),
    contact: str | None = Form(None),
    doctor_notes: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        parsed_scan_date = datetime.strptime(scan_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="scan_date must be in YYYY-MM-DD format.") from exc

    resolution = resolve_or_create_patient(
        db=db,
        provided_patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        contact=contact,
    )
    patient = resolution.patient

    try:
        upload_path = await save_upload_file(file, patient_id=patient.patient_id, scan_date=parsed_scan_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        preprocessed = preprocess_scan(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        analysis = analyze_scan(preprocessed)
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    risk_category = compute_risk_category(
        tumor_detected=analysis.tumor_detected,
        confidence=analysis.confidence,
        tumor_area=analysis.tumor_area,
    )
    uncertainty = confidence_warning(analysis.confidence)

    scan = Scan(
        patient_db_id=patient.id,
        scan_date=parsed_scan_date,
        image_path=upload_path,
        tumor_detected=analysis.tumor_detected,
        tumor_type=analysis.tumor_type,
        confidence_score=analysis.confidence,
        uncertainty_score=analysis.uncertainty_score,
        uncertainty_std=analysis.uncertainty_std,
        tumor_area=analysis.tumor_area,
        tumor_volume=analysis.tumor_volume,
        stage_label=analysis.stage_label,
        stage_confidence=analysis.stage_confidence,
        stage_method=analysis.stage_method,
        risk_category=risk_category,
        explainability_consistency_score=analysis.explainability_consistency_score,
        xai_method=analysis.xai_method,
        model_version=analysis.model_version,
        radiologist_notes=doctor_notes,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    mask_path = save_mask(analysis.mask, scan.id)
    gradcam_path = save_gradcam(analysis.gradcam, scan.id)
    overlay_path = save_overlay(analysis.overlay, scan.id)

    scan.mask_path = mask_path
    scan.gradcam_path = gradcam_path
    scan.overlay_path = overlay_path

    if preprocessed.volume_data is not None and preprocessed.volume_data.ndim == 3:
        manifest_path, _, _ = save_volume_slice_stack(
            preprocessed.volume_data,
            scan_id=scan.id,
            selected_slice_index=preprocessed.selected_slice_index,
        )
        scan.volume_manifest_path = manifest_path

    for class_name, prob in analysis.class_probabilities.items():
        db.add(ScanProbability(scan_id=scan.id, class_name=class_name, probability=float(prob)))
    db.commit()

    previous_scan = db.scalar(
        select(Scan)
        .where(Scan.patient_db_id == patient.id, Scan.id != scan.id)
        .order_by(Scan.scan_date.desc(), Scan.id.desc())
    )

    progression_status = "No previous scan available"
    lti = None
    comparison_row = None

    if previous_scan:
        comp = compare_scans(previous_scan, scan)
        progression_status = comp.progression_status
        lti = comp.longitudinal_tumor_progression_index
        comparison_row = Comparison(
            patient_db_id=patient.id,
            previous_scan_id=comp.previous_scan_id,
            current_scan_id=comp.current_scan_id,
            previous_volume=comp.previous_tumor_volume,
            current_volume=comp.current_tumor_volume,
            absolute_change=comp.absolute_change,
            percentage_change=comp.percentage_change,
            progression_status=comp.progression_status,
            longitudinal_index=comp.longitudinal_tumor_progression_index,
            summary=comp.summary,
        )
        db.add(comparison_row)
        db.commit()
        db.refresh(comparison_row)

    all_scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id)).all()
    chart_path = build_growth_chart(patient_id=patient.patient_id, scans=all_scans)

    report_path = generate_report(
        scan=scan,
        patient=patient,
        comparison=comparison_row,
        progression_chart_path=chart_path,
        class_probabilities=sorted(analysis.class_probabilities.items(), key=lambda item: item[1], reverse=True),
    )
    scan.report_path = report_path
    db.commit()

    rag_text = (
        f"Scan date: {scan.scan_date}. Tumor detected: {scan.tumor_detected}. "
        f"Tumor type: {scan.tumor_type}. Tumor area: {scan.tumor_area:.2f}. "
        f"Tumor volume: {scan.tumor_volume}. Stage: {scan.stage_label}. "
        f"Stage method: {scan.stage_method}. Confidence: {scan.confidence_score:.3f}. "
        f"Uncertainty score: {analysis.uncertainty_score}. "
        f"Risk: {scan.risk_category}. Progression: {progression_status}."
    )
    add_rag_document(db, patient_db_id=patient.id, scan_id=scan.id, document_text=rag_text)

    probs = [
        ClassProbability(class_name=name, probability=float(prob))
        for name, prob in sorted(analysis.class_probabilities.items(), key=lambda item: item[1], reverse=True)
    ]
    volume_slice_urls, selected_slice_index = volume_manifest_to_urls(scan.volume_manifest_path)

    return UploadScanResponse(
        patient_id=patient.patient_id,
        patient_name=patient.name,
        generated_patient_id=resolution.generated_new_id,
        matched_existing_patient=resolution.matched_existing,
        patient_match_strategy=resolution.match_strategy,
        patient_match_score=resolution.match_score,
        scan_id=scan.id,
        scan_date=scan.scan_date,
        tumor_detected=scan.tumor_detected,
        tumor_type=scan.tumor_type,
        confidence_score=scan.confidence_score,
        uncertainty_score=scan.uncertainty_score,
        uncertainty_std=scan.uncertainty_std,
        tumor_area=scan.tumor_area,
        tumor_volume=scan.tumor_volume,
        stage_label=scan.stage_label,
        stage_confidence=scan.stage_confidence,
        stage_method=scan.stage_method,
        volume_units=analysis.volume_units,
        is_area_based_approximation=analysis.is_area_based_approximation,
        risk_category=scan.risk_category,
        uncertainty_warning=uncertainty,
        progression_status=progression_status,
        explainability_consistency_score=scan.explainability_consistency_score,
        explainability_warning=analysis.explainability_warning,
        xai_method=analysis.xai_method,
        longitudinal_tumor_progression_index=lti,
        model_version=scan.model_version,
        runtime_mode=analysis.runtime_mode,
        runtime_note=analysis.runtime_note,
        class_probabilities=probs,
        report_path=report_path,
        gradcam_path=gradcam_path,
        mask_path=mask_path,
        overlay_path=overlay_path,
        image_url=to_storage_url(scan.image_path),
        report_url=f"/api/reports/{scan.id}",
        gradcam_url=to_storage_url(gradcam_path),
        mask_url=to_storage_url(mask_path),
        overlay_url=to_storage_url(overlay_path),
        volume_manifest_url=to_storage_url(scan.volume_manifest_path),
        volume_slice_urls=volume_slice_urls,
        selected_slice_index=selected_slice_index,
        disclaimer=UI_DISCLAIMER,
        stage_note=STAGE_NOTE,
        attribution=ATTRIBUTION,
    )
