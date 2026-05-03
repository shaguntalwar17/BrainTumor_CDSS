# PDF Report Format

Project made by Shagun Talwar

## Header
- Title: AI-Assisted Brain MRI Tumor Analysis Report
- Patient name/ID/age/gender
- Scan date and report generation timestamp

## Mandatory Safety Content
- Medical disclaimer (AI-assisted prototype only)
- Tumor stage limitation statement

## Core Clinical-Demo Sections
1. Detection result (tumor yes/no + confidence)
2. Classification result (predicted class + class probability table)
3. Segmentation summary (mask/overlay + area/volume estimate)
4. Explainability (Grad-CAM + consistency score)
5. Longitudinal comparison (change metrics + progression status)
6. Tumor progression graph
7. Recommendation for expert radiologist review

## Technical Appendix
- Model version
- Risk category
- Confidence warning if low confidence
- Timestamp and inference context notes

## Footer
- Page number
- Project made by Shagun Talwar
