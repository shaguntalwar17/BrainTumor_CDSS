from __future__ import annotations

import argparse

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.models.entities import Comparison, Patient, Scan
from backend.services.comparison_service import build_growth_chart
from backend.services.report_service import generate_report
from backend.utils.config import settings


def main(scan_id: int):
    engine = create_engine(settings.db_url, future=True, connect_args={"check_same_thread": False})
    with Session(engine) as db:
        scan = db.get(Scan, scan_id)
        if not scan:
            raise RuntimeError("Scan not found")

        patient = db.get(Patient, scan.patient_db_id)
        if not patient:
            raise RuntimeError("Patient not found")

        comparison = db.scalar(select(Comparison).where(Comparison.current_scan_id == scan.id))
        all_scans = db.scalars(select(Scan).where(Scan.patient_db_id == patient.id)).all()
        chart_path = build_growth_chart(patient.patient_id, all_scans)

        report_path = generate_report(scan=scan, patient=patient, comparison=comparison, progression_chart_path=chart_path)
        scan.report_path = report_path
        db.commit()

    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF report for a scan")
    parser.add_argument("--scan-id", type=int, required=True)
    args = parser.parse_args()

    main(args.scan_id)
