from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    patient_code: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(128))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(32))
    contact: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    scans: Mapped[list["Scan"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_db_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    scan_date: Mapped[date] = mapped_column(Date)
    image_path: Mapped[str] = mapped_column(String(512))
    mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    gradcam_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tumor_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    tumor_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    tumor_area: Mapped[float] = mapped_column(Float, default=0.0)
    tumor_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_category: Mapped[str] = mapped_column(String(32), default="Low")
    explainability_consistency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), default="prototype-v1")
    radiologist_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    patient: Mapped[Patient] = relationship(back_populates="scans")
    probabilities: Mapped[list["ScanProbability"]] = relationship(
        back_populates="scan",
        cascade="all, delete-orphan",
    )


class ScanProbability(Base):
    __tablename__ = "scan_probabilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), index=True)
    class_name: Mapped[str] = mapped_column(String(64), index=True)
    probability: Mapped[float] = mapped_column(Float, default=0.0)

    scan: Mapped[Scan] = relationship(back_populates="probabilities")


class Comparison(Base):
    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_db_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    previous_scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    current_scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"))
    previous_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    absolute_change: Mapped[float] = mapped_column(Float, default=0.0)
    percentage_change: Mapped[float] = mapped_column(Float, default=0.0)
    progression_status: Mapped[str] = mapped_column(String(64), default="Stable")
    longitudinal_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_db_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id"), index=True)
    document_type: Mapped[str] = mapped_column(String(64), default="scan_summary")
    document_text: Mapped[str] = mapped_column(Text)
    embedding_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    dice: Mapped[float | None] = mapped_column(Float, nullable=True)
    iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    hausdorff95: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_use_case: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
