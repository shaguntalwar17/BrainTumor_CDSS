from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "Brain MRI Tumor AI Platform"
    api_prefix: str = "/api"
    model_runtime_mode: str = "demo"
    classification_model_path: str = ""
    segmentation_model_path: str = ""
    database_url: str | None = None
    db_path: str = "backend/storage/brain_mri.db"
    storage_root: str = "backend/storage"
    output_dir: str = "backend/storage"
    report_dir: str = "backend/storage/reports"
    upload_dir: str = "backend/storage/uploads"
    mask_dir: str = "backend/storage/masks"
    gradcam_dir: str = "backend/storage/gradcam"
    overlay_dir: str = "backend/storage/overlays"
    chart_dir: str = "backend/storage/charts"
    vector_store_dir: str = "backend/storage/vector_store"
    vector_db_dir: str = "backend/storage/vector_store"
    model_artifacts_dir: str = "ml/artifacts"
    low_confidence_threshold: float = 0.55
    medium_risk_area_threshold: float = 2500.0
    high_risk_area_threshold: float = 6000.0
    timezone: str = "Asia/Kolkata"
    cors_origins: str = "*"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_file = Path(self.db_path)
        if not db_file.is_absolute():
            db_file = Path.cwd() / db_file
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_file.as_posix()}"


settings = Settings()
