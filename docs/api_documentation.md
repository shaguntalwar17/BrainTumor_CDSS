# API Documentation

Project made by Shagun Talwar

Base URL (local): `http://127.0.0.1:8000`

## Health
- `GET /`
- `GET /health`
- `GET /api/health`

## Scans
- `POST /upload-scan` (legacy compatibility)
- `POST /api/scans/upload` (primary)
- `GET /scans/{scan_id}` (legacy compatibility)
- `GET /api/scans/{scan_id}` (primary)
- `POST /compare-scans` (legacy compatibility)
- `POST /api/scans/compare` (primary)

## Patients
- `GET /patients` and `GET /api/patients`
- `GET /patients/{patient_id}` and `GET /api/patients/{patient_id}`
- `GET /patients/{patient_id}/scans` and `GET /api/patients/{patient_id}/scans`

## Reports
- `GET /report/{scan_id}` (legacy compatibility)
- `GET /api/reports/{scan_id}` (primary)

## Model Metrics
- `GET /model-metrics` (legacy compatibility)
- `GET /api/models/metrics` (primary)

## RAG
- `POST /rag-query` (legacy compatibility)
- `POST /api/rag/query` (primary)

Supported question patterns:
- "Show summary of Patient XYZ"
- "Compare latest scan with first scan"
- "How has tumor area changed over time?"
- "List previous scan dates"
- "Generate patient history summary"

## Dashboard
- `GET /dashboard/summary` (legacy compatibility)
- `GET /api/dashboard/summary` (primary)

## Response Notes
- Asset preview URLs are exposed via `/storage/...`.
- Patient/scan endpoints include class probability table where available.
- Medical disclaimer and no-stage policy are included in upload responses.
