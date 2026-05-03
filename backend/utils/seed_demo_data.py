from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from backend.database.base import Base
from backend.database.session import SessionLocal, engine
from backend.models.entities import Patient, Scan, ScanProbability
from backend.services.comparison_service import build_growth_chart, compare_scans
from backend.services.inference_service import analyze_scan
from backend.services.preprocessing_service import preprocess_scan
from backend.services.report_service import generate_report
from backend.services.rag_service import add_rag_document
from backend.services.risk_service import compute_risk_category
from backend.services.storage_service import save_gradcam, save_mask, save_overlay


def seed_demo_data() -> None:
    Base.metadata.create_all(bind=engine)

    sample_images = sorted(Path("data/processed").rglob("*.jpg"))
    if len(sample_images) < 2:
        raise RuntimeError("Not enough sample images found under data/processed.")

    with SessionLocal() as db:
        patient = db.scalar(select(Patient).where(Patient.patient_id == "DEMO001"))
        if not patient:
            patient = Patient(
                patient_id="DEMO001",
                patient_code="DEMO001",
                name="Demo Patient",
                age=42,
                gender="Female",
                contact="N/A",
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)

        scan_dates = [date.today() - timedelta(days=120), date.today()]
        created_scans: list[Scan] = []

        for idx, image_path in enumerate(sample_images[:2]):
            pre = preprocess_scan(str(image_path))
            result = analyze_scan(pre)

            scan = Scan(
                patient_db_id=patient.id,
                scan_date=scan_dates[idx],
                image_path=str(image_path).replace("\\", "/"),
                tumor_detected=result.tumor_detected,
                tumor_type=result.tumor_type,
                confidence_score=result.confidence,
                tumor_area=result.tumor_area,
                tumor_volume=result.tumor_volume,
                risk_category=compute_risk_category(result.tumor_detected, result.confidence, result.tumor_area),
                explainability_consistency_score=result.explainability_consistency_score,
                model_version=result.model_version,
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)

            scan.mask_path = save_mask(result.mask, scan.id)
            scan.gradcam_path = save_gradcam(result.gradcam, scan.id)
            scan.overlay_path = save_overlay(result.overlay, scan.id)
            db.commit()

            for class_name, prob in result.class_probabilities.items():
                db.add(ScanProbability(scan_id=scan.id, class_name=class_name, probability=float(prob)))
            db.commit()

            created_scans.append(scan)

        comparison_row = None
        if len(created_scans) >= 2:
            comp = compare_scans(created_scans[0], created_scans[1])
            from backend.models.entities import Comparison

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

        all_scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id)).all()
        chart_path = build_growth_chart(patient_id=patient.patient_id, scans=all_scans)

        for scan in created_scans:
            report_path = generate_report(scan=scan, patient=patient, comparison=comparison_row, progression_chart_path=chart_path)
            scan.report_path = report_path
            db.commit()

            add_rag_document(
                db,
                patient_db_id=patient.id,
                scan_id=scan.id,
                document_text=(
                    f"Demo scan {scan.id} on {scan.scan_date}: tumor={scan.tumor_detected}, "
                    f"type={scan.tumor_type}, area={scan.tumor_area:.2f}, risk={scan.risk_category}."
                ),
            )

    print("Seeded demo patient DEMO001 with sample scans.")


if __name__ == "__main__":
    seed_demo_data()
