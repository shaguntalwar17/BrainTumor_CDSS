# PDF Report Format

Project made by Shagun Talwar and co-developed by Chirag Garg

## Header
- Title: AI-Assisted Brain MRI Tumor Analysis Report
- Patient name/ID/age/gender
- Scan date and report generation timestamp

## Mandatory Safety Content
- Medical disclaimer (AI-assisted prototype only)
- Tumor stage statement: "Tumor stage is not predicted because the uploaded dataset does not contain medically validated staging labels."

## Core Clinical-Demo Sections
1. Detection result (tumor yes/no + confidence)
2. Classification result (predicted class + class probability table)
3. Stage estimate (with method and confidence)
4. Segmentation summary (mask/overlay + area/volume estimate)
5. Explainability (Grad-CAM + consistency score)
6. Longitudinal comparison (change metrics + progression status)
7. Tumor progression graph
8. Recommendation for expert radiologist review

## Technical Appendix
- Model version
- Risk category
- Confidence warning if low confidence
- Stage limitation note (proxy vs validated stage model)
- Timestamp and inference context notes

## Footer
- Page number
- Project made by Shagun Talwar
