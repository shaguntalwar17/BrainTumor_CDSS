from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import Scan


router = APIRouter(tags=["report"])


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

