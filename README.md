# Brain MRI Tumor AI Platform

**Project made by Shagun Talwar**

An advanced academic/research prototype for AI-assisted Brain MRI tumor detection, segmentation, classification, longitudinal progression analysis, explainable AI (Grad-CAM), RAG-grounded patient-history summaries, and professional PDF report generation.

## Medical Disclaimer
This system is an **AI-assisted academic prototype for Brain MRI tumor analysis** and **not** a certified medical diagnostic tool.

- All predictions, segmentations, heatmaps, and reports must be verified by a qualified radiologist or medical professional.
- Tumor stage prediction is not provided because the dataset does not include clinically validated staging labels.
- 2D MRI analysis uses area-based approximation, not true clinical volumetric diagnosis.

## 1. Problem Statement
Brain tumor analysis is often fragmented across tools. This project provides a single end-to-end platform that demonstrates:

- MRI upload and preprocessing verification
- Tumor detection + segmentation + classification
- Explainability overlays (Grad-CAM)
- Patient history and longitudinal scan comparison
- Risk and progression interpretation
- Professional report generation
- RAG-grounded summary retrieval from prior records

## 2. Objectives
- Build modular pipelines for detection, segmentation, classification, explainability, longitudinal comparison, report generation, and retrieval.
- Maintain transparent limitations and uncertainty messaging.
- Provide a premium web dashboard suitable for final-year/research demo and viva.

## 3. Features
- Detection: Tumor present/absent
- Segmentation: Binary mask, overlay, boundary, area estimation
- Classification: Multi-class tumor type prediction
- Explainability: Grad-CAM with consistency score vs segmentation
- Longitudinal: Change tracking across scans with progression status
- Risk category: Low/Medium/High based on transparent logic
- LTI novelty: Longitudinal Tumor Progression Index (0-100)
- RAG novelty: Grounded patient-history retrieval from stored summaries
- PDF report: Structured, hospital-style report with disclaimer
- Model leaderboard: Classification + segmentation metrics view

## 4. Architecture
```text
[Frontend: Next.js + Tailwind + Framer Motion]
                |
                v
[FastAPI Backend]
  |-- Preprocessing Verification
  |-- Detection / Segmentation / Classification / Grad-CAM
  |-- Risk + Longitudinal Comparison Engine
  |-- Report Generator (PDF)
  |-- RAG Query Service
  |
  v
[SQLite/PostgreSQL via SQLAlchemy] + [Vector Index: ChromaDB fallback TF-IDF]
  |
  v
[Storage: MRI files, masks, Grad-CAM, overlays, reports, charts]
```

## 5. Model Pipeline
1. Image loading (JPG/PNG + optional NIfTI/DICOM)
2. Resize, normalize, denoise, skull-strip support
3. Detection and classification
4. Segmentation mask and area/volume estimate
5. Grad-CAM heatmap and explainability consistency score
6. Risk category + uncertainty warning
7. Longitudinal comparison with previous scan
8. RAG document creation and indexing
9. PDF report generation

## 6. Dataset Details
Current repository includes image folders under `data/raw` and `data/processed`.

Expected segmentation masks should be provided in:
- `data/masks/train`
- `data/masks/test`

For custom datasets, update paths in:
- `ml/configs/classification_config.yaml`
- `ml/configs/segmentation_config.yaml`

## 7. Preprocessing Steps
Implemented in backend and ML scripts:
- Loading and format detection
- Intensity normalization
- Gaussian denoise
- Skull-strip approximation (2D)
- Resize to model input
- Optional NIfTI and DICOM loaders

Validation script:
- `ml/preprocessing/verify_dataset.py`

Checks:
- missing/zero-byte files
- corrupt images
- duplicates
- duplicate inferred patient IDs
- inconsistent dimensions by label

## 8. Segmentation Training
Script: `ml/train_segmentation.py`

Supported model options:
- `unet`
- `attention_unet`
- `unet++` (requires `segmentation_models_pytorch`)
- `monai_unet` / `unetr` / `swinunetr` (requires MONAI)

Loss options:
- `dice_bce`
- `dice_focal`

Saved outputs:
- best checkpoint
- training history JSON
- metrics JSON
- if true masks are missing, script can run in clearly labeled pseudo-mask demo mode

## 9. Classification Training
Script: `ml/train_classification.py`

Includes mentor-requested ResNeXt101 pipeline:
- pretrained ResNeXt101 backbone
- MLP classification head (Linear + BN + ReLU + Dropout + Linear)
- phase 1: freeze backbone, train head
- phase 2: unfreeze last blocks and fine-tune
- weighted sampler and weighted loss for class imbalance
- scheduler, early stopping, optional mixed precision

Comparison-ready model factory includes:
- ResNeXt101
- ResNet50
- DenseNet121
- EfficientNet-B3
- ConvNeXt-Tiny

## 10. Evaluation Metrics
Classification script: `ml/evaluate_classification.py`
- accuracy
- precision/recall/F1
- ROC-AUC OVR (if computable)
- confusion matrix
- confidence distribution

Segmentation script: `ml/evaluate_segmentation.py`
- Dice
- IoU
- precision
- recall/sensitivity
- specificity
- sample prediction panels

Note: Hausdorff95 is reserved for future extension (placeholder field retained).

## 11. Longitudinal Comparison
API and script compare previous/current scans and compute:
- absolute and percentage change
- progression status:
  - New tumor detected
  - Tumor no longer detected
  - Significantly increased
  - Slightly increased
  - Decreased
  - Stable
  - No Tumor
- confidence difference
- tumor type change
- longitudinal tumor progression index

## 12. RAG / Vector DB
- Raw MRI/masks/reports remain in filesystem storage.
- Structured metadata stored in SQL DB.
- Summaries stored in `rag_documents`.
- Vector index build script: `ml/build_vector_index.py`
  - tries ChromaDB
  - falls back to TF-IDF index JSON

RAG query API: `POST /rag-query`
- grounded strictly on stored patient records
- includes citations (`scan_id`, `scan_date`)

## 13. Professional Report Generation
- Report generator: `backend/services/report_service.py`
- Includes:
  - patient information
  - tumor detection/classification
  - segmentation summary
  - area/volume estimate
  - risk category
  - Grad-CAM section
  - longitudinal summary
  - progression graph
  - model/version and limitations
  - recommendation + disclaimer
  - footer attribution: Project made by Shagun Talwar

Script: `ml/generate_report.py --scan-id <id>`

## 14. Repository Structure
```text
backend/
frontend/
ml/
docs/
reports/
sample_outputs/
notebooks/
docker-compose.yml
README.md
```

Core docs:
- `docs/architecture.md`
- `docs/model_pipeline.md`
- `docs/api_documentation.md`
- `docs/report_format.md`
- `docs/demo_script.md`

## 15. API Endpoints
- `POST /api/scans/upload` (primary) and `POST /upload-scan` (compatibility)
- `GET /api/patients`
- `GET /api/patients/{patient_id}`
- `GET /api/patients/{patient_id}/scans`
- `GET /api/scans/{scan_id}`
- `POST /api/scans/compare` (primary) and `POST /compare-scans` (compatibility)
- `GET /api/reports/{scan_id}` (primary) and `GET /report/{scan_id}` (compatibility)
- `GET /api/models/metrics` (primary) and `GET /model-metrics` (compatibility)
- `POST /api/rag/query` (primary) and `POST /rag-query` (compatibility)
- `GET /api/dashboard/summary`

Detailed API guide: `docs/api_documentation.md`

## 16. Setup Instructions
### Prerequisites
- Python 3.10+
- Node 18+
- (Optional) CUDA GPU

### Backend setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Optional config:
```bash
copy .env.example .env
```

### Frontend setup
```bash
cd frontend
npm install
npm run dev
```

Default URLs:
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://localhost:3000`

### Docker (optional)
```bash
docker compose up --build
```

### Demo sample data (optional)
```bash
python backend/utils/seed_demo_data.py
```
This creates a demo patient (`DEMO001`) with sample scans for viva presentation flow.

## 17. Train Models
```bash
python ml/train_classification.py
python ml/train_segmentation.py
```

Example advanced commands:
```bash
python ml/classification/train.py --config ml/configs/classification_config.yaml
python ml/segmentation/train.py --config ml/configs/segmentation_config.yaml
python ml/classification/train.py --data data/processed --epochs 20 --output models/classification/resnext101_best.pth
python ml/segmentation/train.py --images data/segmentation/images --masks data/segmentation/masks --epochs 20 --output models/segmentation/unet_best.pth
```

## 18. Evaluate Models
```bash
python ml/evaluate_classification.py
python ml/evaluate_segmentation.py
```

## 19. Runtime Modes
- `MODEL_RUNTIME_MODE=demo`
  - Uses baseline/demo inference behavior without trained checkpoints.
  - Outputs are clearly marked as non-clinical prototype results.
- `MODEL_RUNTIME_MODE=trained`
  - Requires valid checkpoint paths:
    - `CLASSIFICATION_MODEL_PATH`
    - `SEGMENTATION_MODEL_PATH`
  - Missing checkpoint behavior:
    - `Trained model checkpoint not found. Please train the model or switch MODEL_RUNTIME_MODE=demo.`

## 20. Generate Explainability and Reports
```bash
python ml/generate_gradcam.py --image <path_to_mri>
python ml/generate_report.py --scan-id <scan_id>
python ml/run_inference.py --image <path_to_mri>
python ml/build_vector_index.py
python ml/scripts/progress_parser.py --log logs/classification_train.log
python ml/scripts/progress_parser.py --log logs/classification_train.log --watch --interval 15
```

## 21. Screenshot Placeholders
Add screenshots here for final demo:
- `docs/screenshots/landing.png`
- `docs/screenshots/upload.png`
- `docs/screenshots/dashboard.png`
- `docs/screenshots/history.png`
- `docs/screenshots/comparison.png`
- `docs/screenshots/models.png`

## 22. Documentation
- `docs/architecture.md`
- `docs/model_pipeline.md`
- `docs/api_documentation.md`
- `docs/report_format.md`
- `docs/demo_script.md`
- `docs/DEMO_FLOW.md`

## 23. Limitations
- Prototype inference defaults are heuristic if trained weights are not provided.
- Advanced 3D volumetric precision depends on full modality-aware datasets.
- DICOM/NIfTI paths require tested clinical-grade pipelines for deployment.
- This system is not regulatory-certified for diagnosis.

## 24. Future Scope
- Full nnU-Net pipeline and automated architecture selection
- Better multi-modal fusion (T1/T1CE/T2/FLAIR)
- Calibrated uncertainty estimation and Bayesian inference
- PACS/RIS integration for hospital workflow simulation
- Federated and continual learning for longitudinal adaptation

---

### Attribution
**Project made by Shagun Talwar**
