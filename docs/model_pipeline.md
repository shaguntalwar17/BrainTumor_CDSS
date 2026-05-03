# Model Pipeline

Project made by Shagun Talwar

## 1. Preprocessing
- File format support: JPG/PNG, optional NIfTI/DICOM.
- Steps: normalization, denoise, skull-strip approximation, resize.
- Validation scripts check missing/corrupt files, duplicates, and dimensional mismatch.

## 2. Detection + Classification
- Main model target: ResNeXt101 transfer-learning workflow.
- Two-stage fine-tuning:
1. Freeze backbone, train MLP head.
2. Unfreeze last blocks, fine-tune with lower LR.
- Comparative model options: ResNet50, DenseNet121, EfficientNet-B3, ConvNeXt-Tiny.

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
- Generates patient progression chart over scan dates.
- Uses area-based approximation for 2D scans.

## 6. Report + Retrieval
- PDF combines detection/classification/segmentation/explainability/comparison outputs.
- RAG uses stored textual scan summaries and citations to answer patient-history queries.
- Retrieval is grounded only in stored records; no synthetic medical history.
