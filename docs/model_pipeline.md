# Model Pipeline

Project made by Shagun Talwar and co-developed by Chirag Garg

## 1. Preprocessing
- File format support: JPG/PNG, optional NIfTI/DICOM.
- Steps: normalization, denoise, skull-strip approximation, resize.
- Validation scripts check missing/corrupt files, duplicates, and dimensional mismatch.

## 2. Detection + Classification
- Main model target: ResNeXt101 transfer-learning workflow.
- Inference hardening: optional ensemble checkpoints + test-time augmentation + temperature scaling.
- Class calibration: prior-weight adjustment (`CLASSIFICATION_PRIOR_WEIGHTS`) to reduce class bias drift.
- Detection gate: morphology + classifier confidence fusion (`tumor_detection_threshold`, `no_tumor_probability_threshold`).
- Pituitary false-positive guard: re-checks predicted pituitary cases using segmented area/centroid constraints before finalizing class.
- Two-stage fine-tuning:
1. Freeze backbone, train MLP head.
2. Unfreeze last blocks, fine-tune with lower LR.
- Comparative model options: ResNet50, DenseNet121, EfficientNet-B3, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16.

## 3. Segmentation
- Implemented/ready options:
1. U-Net
2. Attention U-Net
3. U-Net++
4. MONAI UNet / UNETR / SwinUNETR
- Training losses: Dice+BCE or Dice+Focal.
- Metrics: Dice, IoU, precision, recall, specificity.

## 4. Explainability
- Grad-CAM heatmap is generated for each scan.
- Explainability Consistency Score compares Grad-CAM attention vs predicted segmentation mask overlap.
- Low-overlap scenarios are flagged as reliability warnings.

## 5. Longitudinal Comparison
- Compares current scan against previous scan(s).
- Computes absolute/percentage change, progression status, and progression index.
- Progression labels include: Improved, Stable, Slightly increased, Significantly increased, New tumor detected, Tumor no longer detected.
- Generates patient progression chart over scan dates.
- Generates a growth/reduction map from previous vs current segmentation masks when both masks are available.
- Uses area-based approximation for 2D scans.

## 6. Report + Retrieval
- PDF combines detection/classification/segmentation/explainability/comparison outputs.
- RAG uses stored textual scan summaries and citations to answer patient-history queries.
- Retrieval is grounded only in stored patient records; no synthetic medical history.

## 7. Stage Estimation (Research)
- Preferred: trained stage model checkpoint (Low-grade vs High-grade proxy workflow).
- Fallback: transparent proxy burden rules based on tumor area/volume and tumor type.
- Stage output is explicitly marked as research-only in UI and PDF report.
