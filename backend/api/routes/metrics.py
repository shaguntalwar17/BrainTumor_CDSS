from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.models.entities import ModelMetric
import json

from backend.services.metrics_service import seed_sample_metrics_if_empty, sync_metrics_from_json, upsert_metric_row


router = APIRouter(tags=["metrics"])


@router.get("/model-metrics")
@router.get("/api/models/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    seed_sample_metrics_if_empty(db)
    metrics_file = Path("ml/artifacts/model_metrics.json")
    if metrics_file.exists():
        sync_metrics_from_json(db, str(metrics_file))

    for single_file in [
        Path("ml/artifacts/classification/classification_metrics.json"),
        Path("ml/artifacts/segmentation/segmentation_metrics.json"),
        Path("ml/artifacts/segmentation/segmentation_eval_metrics.json"),
    ]:
        if single_file.exists():
            try:
                payload = json.loads(single_file.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    upsert_metric_row(db, payload)
            except Exception:
                pass

    rows = db.scalars(select(ModelMetric).order_by(ModelMetric.task_type.asc(), ModelMetric.model_name.asc())).all()
    return {
        "note": "Values marked null mean metrics not computed yet. Run training/evaluation scripts to populate actual results.",
        "items": [
            {
                "id": r.id,
                "model_name": r.model_name,
                "task_type": r.task_type,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "auc": r.auc,
                "dice": r.dice,
                "iou": r.iou,
                "hausdorff95": r.hausdorff95,
                "inference_time": r.inference_time,
                "training_time": r.training_time,
                "model_size": r.model_size,
                "best_use_case": r.best_use_case,
                "status": (
                    "real"
                    if any(v is not None for v in [r.accuracy, r.f1_score, r.auc, r.dice, r.iou])
                    else "demo"
                ),
            }
            for r in rows
        ],
    }
