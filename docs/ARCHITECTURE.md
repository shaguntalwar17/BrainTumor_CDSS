# System Architecture

Project made by Shagun Talwar

## 1. High-Level Flow

```text
Frontend (Next.js + Tailwind + Framer Motion)
          |
          v
FastAPI API Layer
  |- /upload-scan
  |- /patients, /patients/{id}, /patients/{id}/scans
  |- /compare-scans
  |- /report/{scan_id}
  |- /model-metrics
  |- /rag-query
          |
          v
Service Layer
  |- preprocessing_service
  |- inference_service (detection/classification/segmentation/xai)
  |- comparison_service (longitudinal stats + progression chart + growth map)
  |- report_service (PDF)
  |- rag_service (patient-history grounded retrieval)
          |
          v
Persistence Layer
  |- SQL database (SQLite/PostgreSQL style schema)
  |- File/object storage paths for MRI/mask/gradcam/overlay/report/chart assets
  |- Vector index from stored textual summaries (ChromaDB with TF-IDF fallback)
```

## 2. Data Storage Strategy
- Raw MRI images, masks, overlays, Grad-CAM outputs, and PDFs are stored in filesystem/object storage paths.
- Structured metadata and longitudinal fields are stored in SQL tables (`patients`, `scans`, `comparisons`, `rag_documents`, `model_metrics`).
- Vector index stores only textual summaries and references, never raw MRI pixels.

## 3. Inference Runtime Modes
- `demo`: Research-safe heuristic fallback for showcase when checkpoints are unavailable.
- `trained`: Uses trained classification and segmentation checkpoints for actual model inference.

## 4. Longitudinal Pipeline
1. Identify previous scan for same patient.
2. Compare area/volume, confidence, tumor type, and risk.
3. Compute progression status and longitudinal tumor progression index.
4. Generate trend chart and optional mask growth/reduction visualization.
5. Store comparison row and include summary in report + RAG document.

## 5. Safety and Clinical Boundaries
- Explicit disclaimer in UI and PDF report.
- Tumor stage is intentionally not predicted (no medically validated stage labels in dataset).
- Low-confidence outputs trigger warning messaging.
- RAG answers are grounded strictly in stored patient records with citations.
