from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    project_name: str = "Brain MRI Tumor AI Platform"
    api_prefix: str = "/api"
    model_runtime_mode: str = "auto"
    classification_model_path: str = "ml/artifacts/classification_resnet50_balanced/best_classification_resnet50_balanced.pt"
    classification_ensemble_paths: str = ""
    segmentation_model_path: str = "ml/artifacts/segmentation/best_segmentation.pt"
    stage_model_path: str = ""
    stage_model_name: str = "resnet18"
    stage_classes: str = "Low-grade,High-grade"
    stage_proxy_enabled: bool = True
    classification_temperature: float = 1.0
    classification_prior_weights: str = "Glioma=1.0,Meningioma=1.0,Pituitary=0.86,No_Tumor=1.08"
    enable_tta_inference: bool = True
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
    volume_preview_dir: str = "backend/storage/volumes"
    vector_store_dir: str = "backend/storage/vector_store"
    vector_db_dir: str = "backend/storage/vector_store"
    model_artifacts_dir: str = "ml/artifacts"
    low_confidence_threshold: float = 0.55
    medium_risk_area_threshold: float = 2500.0
    high_risk_area_threshold: float = 6000.0
    mc_dropout_samples: int = 10
    mc_dropout_rate: float = 0.35
    tumor_detection_threshold: float = 0.45
    no_tumor_probability_threshold: float = 0.56
    min_tumor_area_ratio: float = 0.004
    min_tumor_pixels: int = 260
    pituitary_guard_enabled: bool = True
    pituitary_guard_margin: float = 0.12
    pituitary_area_ratio_max: float = 0.16
    pituitary_centroid_x_min: float = 0.28
    pituitary_centroid_x_max: float = 0.72
    pituitary_centroid_y_min: float = 0.40
    pituitary_centroid_y_max: float = 0.90
    xai_method: str = "hirescam"
    nifti_modality_index: int = 0
    skull_strip_mode: str = "otsu"
    hd_bet_command: str = ""
    rag_guidelines_path: str = "docs/WHO_CNS_TUMOR_GUIDELINES.md"
    include_guideline_rag: bool = False
    timezone: str = "Asia/Kolkata"
    cors_origins: str = "*"

    def _resolve_to_repo_root(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return raw
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve().as_posix()

    def model_post_init(self, __context) -> None:
        path_fields = [
            "db_path",
            "storage_root",
            "output_dir",
            "report_dir",
            "upload_dir",
            "mask_dir",
            "gradcam_dir",
            "overlay_dir",
            "chart_dir",
            "volume_preview_dir",
            "vector_store_dir",
            "vector_db_dir",
            "model_artifacts_dir",
            "rag_guidelines_path",
            "classification_model_path",
            "segmentation_model_path",
            "stage_model_path",
        ]
        for field in path_fields:
            val = getattr(self, field)
            setattr(self, field, self._resolve_to_repo_root(val))

        if self.classification_ensemble_paths:
            normalized: list[str] = []
            for raw in self.classification_ensemble_paths.split(","):
                item = raw.strip()
                if not item:
                    continue
                normalized.append(self._resolve_to_repo_root(item))
            self.classification_ensemble_paths = ",".join(normalized)

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_file.as_posix()}"


settings = Settings()
