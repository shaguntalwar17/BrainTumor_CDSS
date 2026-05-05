from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models.entities import ModelMetric


SAMPLE_METRICS_NOTE = "Sample/demo metrics. Replace by actual training outputs from ml/evaluation scripts."


def seed_sample_metrics_if_empty(db: Session) -> None:
    existing = db.scalar(select(ModelMetric.id).limit(1))
    if existing is not None:
        return

    samples = [
        {
            "model_name": "ResNeXt101_32x8d",
            "task_type": "classification",
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "auc": None,
            "inference_time": None,
            "model_size": None,
            "training_time": None,
            "best_use_case": "Primary mentor-recommended backbone. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "ResNet50",
            "task_type": "classification",
            "best_use_case": "Baseline comparison. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "DenseNet121",
            "task_type": "classification",
            "best_use_case": "Efficient medical-image baseline. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "EfficientNet-B3",
            "task_type": "classification",
            "best_use_case": "Balanced speed/accuracy. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "EfficientNetV2-S",
            "task_type": "classification",
            "best_use_case": "Modern efficient CNN for faster convergence. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "ConvNeXt-Tiny",
            "task_type": "classification",
            "best_use_case": "Modern CNN benchmark. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "ViT-B/16",
            "task_type": "classification",
            "best_use_case": "Transformer comparison. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "U-Net",
            "task_type": "segmentation",
            "dice": None,
            "iou": None,
            "hausdorff95": None,
            "best_use_case": "Segmentation baseline. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "Attention U-Net",
            "task_type": "segmentation",
            "dice": None,
            "iou": None,
            "hausdorff95": None,
            "best_use_case": "Focus on salient regions. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "U-Net++",
            "task_type": "segmentation",
            "dice": None,
            "iou": None,
            "hausdorff95": None,
            "best_use_case": "Dense skip refinement. " + SAMPLE_METRICS_NOTE,
        },
        {
            "model_name": "SwinUNETR",
            "task_type": "segmentation",
            "dice": None,
            "iou": None,
            "hausdorff95": None,
            "best_use_case": "Advanced MONAI transformer option. " + SAMPLE_METRICS_NOTE,
        },
    ]

    for item in samples:
        db.add(ModelMetric(**item))
    db.commit()


def sync_metrics_from_json(db: Session, metrics_json_path: str) -> None:
    path = Path(metrics_json_path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        return

    for row in payload:
        model_name = row.get("model_name")
        task_type = row.get("task_type")
        if not model_name or not task_type:
            continue
        existing = db.scalar(
            select(ModelMetric).where(
                func.lower(ModelMetric.model_name) == str(model_name).lower(),
                ModelMetric.task_type == task_type,
            )
        )
        if existing:
            for key, val in row.items():
                if hasattr(existing, key):
                    setattr(existing, key, val)
        else:
            db.add(ModelMetric(**row))
    db.commit()


def upsert_metric_row(db: Session, row: dict) -> None:
    model_name = row.get("model_name")
    task_type = row.get("task_type")
    if not model_name or not task_type:
        return

    existing = db.scalar(
        select(ModelMetric).where(
            func.lower(ModelMetric.model_name) == str(model_name).lower(),
            ModelMetric.task_type == task_type,
        )
    )

    def _metric_score(payload: dict, kind: str) -> float:
        if kind == "classification":
            f1 = payload.get("f1_score")
            auc = payload.get("auc")
            acc = payload.get("accuracy")
            values = [float(v) for v in [f1, auc, acc] if v is not None]
            return values[0] if values else -1.0
        dice = payload.get("dice")
        iou = payload.get("iou")
        values = [float(v) for v in [dice, iou] if v is not None]
        return values[0] if values else -1.0

    if existing:
        existing_payload = {
            "accuracy": existing.accuracy,
            "f1_score": existing.f1_score,
            "auc": existing.auc,
            "dice": existing.dice,
            "iou": existing.iou,
        }
        existing_score = _metric_score(existing_payload, task_type)
        incoming_score = _metric_score(row, task_type)

        if incoming_score >= existing_score:
            for key, val in row.items():
                if hasattr(existing, key):
                    setattr(existing, key, val)
        else:
            # Preserve better current metrics; fill only missing fields from incoming payload.
            for key, val in row.items():
                if not hasattr(existing, key):
                    continue
                if getattr(existing, key) is None and val is not None:
                    setattr(existing, key, val)
    else:
        db.add(ModelMetric(**row))
    db.commit()
