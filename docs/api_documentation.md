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
- `POST /scans/{scan_id}/correct-mask` (legacy compatibility)
- `POST /api/scans/{scan_id}/correct-mask` (primary)

Upload notes:
- `patient_id` is optional. If omitted, backend auto-generates a unique patient ID.
- Backend also attempts similarity-based matching to existing patients using profile signature.
- Response includes `generated_patient_id`, `matched_existing_patient`, `patient_match_strategy`, and `patient_match_score`.

## Patients
- `GET /patients` and `GET /api/patients`
- `GET /patients/{patient_id}` and `GET /api/patients/{patient_id}`
- `GET /patients/{patient_id}/scans` and `GET /api/patients/{patient_id}/scans`

## Reports
- `GET /report/{scan_id}` (legacy compatibility)
- `GET /api/reports/{scan_id}` (primary)
- `GET /reports` and `GET /api/reports` list generated reports
- `GET /reports/comparison/{patient_id}/{previous_scan_id}/{current_scan_id}`
- `GET /api/reports/comparison/{patient_id}/{previous_scan_id}/{current_scan_id}`

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

RAG safety policy:
- Responses are grounded strictly in stored patient records for the requested patient ID.
- Citations include scan IDs and scan dates from persisted records.

## Dashboard
- `GET /dashboard/summary` (legacy compatibility)
- `GET /api/dashboard/summary` (primary)

## Response Notes
- Asset preview URLs are exposed via `/storage/...`.
- 3D uploads include `volume_manifest_url`, `volume_slice_urls`, and `selected_slice_index` for interactive slice scrolling.
- Patient/scan endpoints include class probability table where available.
- Upload responses include uncertainty fields (`uncertainty_score`, `uncertainty_std`) and explainability method metadata.
- Upload and scan responses include stage fields (`stage_label`, `stage_confidence`, `stage_method`).
- Medical disclaimer and no-stage policy are included in upload responses.
- Comparison responses additionally include `growth_map_url` when both previous and current masks are available.
