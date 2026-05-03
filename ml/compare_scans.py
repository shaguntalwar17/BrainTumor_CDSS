from __future__ import annotations

import argparse
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.models.entities import Scan
from backend.services.comparison_service import compare_scans
from backend.utils.config import settings


def main(previous_scan_id: int, current_scan_id: int):
    engine = create_engine(settings.db_url, future=True, connect_args={"check_same_thread": False})
    with Session(engine) as db:
        previous = db.get(Scan, previous_scan_id)
        current = db.get(Scan, current_scan_id)
        if not previous or not current:
            raise RuntimeError("Scan ID(s) not found")

        result = compare_scans(previous, current)
        payload = {
            "previous_scan_id": result.previous_scan_id,
            "current_scan_id": result.current_scan_id,
            "previous_scan_date": str(result.previous_scan_date),
            "current_scan_date": str(result.current_scan_date),
            "previous_tumor_area": result.previous_tumor_area,
            "current_tumor_area": result.current_tumor_area,
            "absolute_change": result.absolute_change,
            "percentage_change": result.percentage_change,
            "progression_status": result.progression_status,
            "longitudinal_tumor_progression_index": result.longitudinal_tumor_progression_index,
            "summary": result.summary,
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two scans")
    parser.add_argument("--previous-scan-id", type=int, required=True)
    parser.add_argument("--current-scan-id", type=int, required=True)
    args = parser.parse_args()

    main(args.previous_scan_id, args.current_scan_id)
