from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import compare, dashboard, health, metrics, patients, rag, report, scans, upload
from backend.database.base import Base
from backend.database.migrations import apply_lightweight_migrations
from backend.database.session import SessionLocal, engine
from backend.services.metrics_service import seed_sample_metrics_if_empty
from backend.utils.config import settings
from backend.utils.pathing import ensure_dir


app = FastAPI(title=settings.project_name, version="1.0.0")

# Ensure tables are available even in script/test contexts where startup hooks may be skipped.
Base.metadata.create_all(bind=engine)
apply_lightweight_migrations(engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins == "*" else [settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_dir(settings.storage_root)
app.mount("/storage", StaticFiles(directory=settings.storage_root), name="storage")

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(patients.router)
app.include_router(compare.router)
app.include_router(report.router)
app.include_router(metrics.router)
app.include_router(rag.router)
app.include_router(scans.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    apply_lightweight_migrations(engine)
    with SessionLocal() as db:
        seed_sample_metrics_if_empty(db)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
