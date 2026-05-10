# Stage Dataset Guide (Research)

Project made by Shagun Talwar and co-developed by Chirag Garg

## Goal
Enable research-stage estimation for glioma burden by training a stage/grade model on MRI slices.

## Practical Public Data Strategy
There is no universally accepted public MRI dataset with full clinically validated stage labels for all tumor types used in simple 2D tumor-classification projects.

Best practical alternative implemented in this project:
- Use **TCGA-LGG** (lower-grade glioma) and **TCGA-GBM** (glioblastoma / high-grade) MRI collections as a binary grade proxy dataset.
- Train a **Low-grade vs High-grade** stage model and plug it into inference through `STAGE_MODEL_PATH`.

## Credible Sources
- TCGA-GBM collection (TCIA): https://www.cancerimagingarchive.net/collection/tcga-gbm/
- TCGA-LGG collection (TCIA): https://www.cancerimagingarchive.net/collection/tcga-lgg/
- BRATS benchmark paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4833122/
- TCGA glioma MRI labels/radiomics release: https://www.nature.com/articles/sdata2017117

## Folder Structure for Stage Training
Expected by `ml/configs/stage_config.yaml`:

```text
data/stage/
  train/
    Low-grade/
    High-grade/
  val/
    Low-grade/
    High-grade/
```

## Training Commands
```bash
python ml/train_stage.py
python ml/evaluate_stage.py
```

## Runtime Integration
Set environment variables:

```bash
STAGE_MODEL_PATH=ml/artifacts/stage/best_stage.pt
STAGE_MODEL_NAME=resnet18
STAGE_CLASSES=Low-grade,High-grade
STAGE_PROXY_ENABLED=true
```

## Limitation Disclosure
- This is **grade proxy estimation**, not definitive clinical staging.
- Stage output in the app/report is explicitly marked research-only and must be verified by a certified radiologist/oncologist.
