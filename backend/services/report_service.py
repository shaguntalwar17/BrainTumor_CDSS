from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.models.entities import Comparison, Patient, Scan
from backend.services.risk_service import confidence_warning
from backend.utils.config import settings
from backend.utils.disclaimer import ATTRIBUTION, REPORT_DISCLAIMER, STAGE_NOTE
from backend.utils.pathing import ensure_dir


class _NumberedCanvasMixin:
    def __call__(self, canvas, doc):
        page_num = canvas.getPageNumber()
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 1.2 * cm, f"Page {page_num}")
        canvas.drawRightString(19.3 * cm, 1.2 * cm, ATTRIBUTION)
        canvas.restoreState()


def _kv_table(data: list[tuple[str, str]]) -> Table:
    table = Table(data, colWidths=[6 * cm, 10.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102A43")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD2D9")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def generate_report(
    scan: Scan,
    patient: Patient,
    comparison: Comparison | None,
    progression_chart_path: str | None,
    class_probabilities: list[tuple[str, float]] | None = None,
) -> str:
    ensure_dir(settings.report_dir)
    out_path = Path(settings.report_dir) / f"scan_{scan.id}_report.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#102A43"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        parent=styles["BodyText"],
        fontSize=9,
        textColor=colors.HexColor("#334E68"),
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.8 * cm,
    )

    content = []
    content.append(Paragraph("AI-Assisted Brain MRI Tumor Analysis Report", title_style))
    content.append(Paragraph("Academic/Research Prototype Report", subtitle_style))
    content.append(Paragraph(f"<b>Disclaimer:</b> {REPORT_DISCLAIMER}", styles["BodyText"]))
    content.append(Paragraph(f"<b>Limitation:</b> {STAGE_NOTE}", styles["BodyText"]))
    content.append(Spacer(1, 0.3 * cm))

    uncertainty = confidence_warning(scan.confidence_score)
    info_rows = [
        ("Field", "Value"),
        ("Patient Name", patient.name),
        ("Patient ID", patient.patient_id),
        ("Age", str(patient.age)),
        ("Gender", patient.gender),
        ("Scan Date", str(scan.scan_date)),
        ("Report Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
        ("Tumor Detected", "Yes" if scan.tumor_detected else "No"),
        ("Tumor Type", scan.tumor_type or "Not Detected"),
        ("Tumor Area (pixels)", f"{scan.tumor_area:.2f}"),
        ("Tumor Volume Estimate", f"{scan.tumor_volume:.2f}" if scan.tumor_volume is not None else "Area-based approximation"),
        ("Confidence Score", f"{scan.confidence_score:.3f}"),
        ("Risk Category", scan.risk_category),
        ("Uncertainty Warning", uncertainty or "None"),
        ("Explainability Consistency", f"{(scan.explainability_consistency_score or 0):.3f}"),
        ("Model Version", scan.model_version),
    ]
    content.append(_kv_table(info_rows))
    content.append(Spacer(1, 0.4 * cm))

    content.append(Paragraph("Tumor Classification Result", styles["Heading3"]))
    if class_probabilities:
        prob_rows = [("Class", "Probability (%)")]
        for class_name, prob in class_probabilities:
            prob_rows.append((class_name, f"{prob * 100:.2f}"))
        content.append(_kv_table(prob_rows))
    else:
        content.append(Paragraph("Class probability table unavailable for this scan context.", styles["BodyText"]))
    content.append(Spacer(1, 0.2 * cm))

    if scan.image_path and Path(scan.image_path).exists():
        content.append(Paragraph("Uploaded MRI Preview", styles["Heading3"]))
        content.append(Image(scan.image_path, width=7.0 * cm, height=7.0 * cm))
        content.append(Spacer(1, 0.2 * cm))

    if scan.overlay_path and Path(scan.overlay_path).exists():
        content.append(Paragraph("Segmentation + Explainability Overlay", styles["Heading3"]))
        content.append(Image(scan.overlay_path, width=7.0 * cm, height=7.0 * cm))
        content.append(Spacer(1, 0.2 * cm))

    if scan.gradcam_path and Path(scan.gradcam_path).exists():
        content.append(Paragraph("Grad-CAM Heatmap", styles["Heading3"]))
        content.append(Image(scan.gradcam_path, width=7.0 * cm, height=7.0 * cm))
        content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("Longitudinal Comparison", styles["Heading3"]))
    if comparison:
        cmp_rows = [
            ("Field", "Value"),
            ("Previous Scan ID", str(comparison.previous_scan_id)),
            ("Current Scan ID", str(comparison.current_scan_id)),
            ("Absolute Change", f"{comparison.absolute_change:.2f}"),
            ("Percentage Change", f"{comparison.percentage_change:.2f}%"),
            ("Progression Status", comparison.progression_status),
            ("Longitudinal Tumor Progression Index", f"{(comparison.longitudinal_index or 0):.2f}/100"),
            ("Summary", comparison.summary),
        ]
        content.append(_kv_table(cmp_rows))
    else:
        content.append(Paragraph("No previous scan available for comparison.", styles["BodyText"]))

    if progression_chart_path and Path(progression_chart_path).exists():
        content.append(Spacer(1, 0.2 * cm))
        content.append(Paragraph("Tumor Progression Graph", styles["Heading3"]))
        content.append(Image(progression_chart_path, width=14.5 * cm, height=6 * cm))

    content.append(Spacer(1, 0.2 * cm))
    content.append(Paragraph("AI Summary", styles["Heading3"]))
    summary_text = (
        f"Based on stored model outputs, this scan indicates tumor detection = {scan.tumor_detected}, "
        f"type = {scan.tumor_type or 'Not detected'}, confidence = {scan.confidence_score:.3f}, "
        f"risk = {scan.risk_category}, and area = {scan.tumor_area:.2f} pixels."
    )
    if comparison:
        summary_text += (
            f" Compared with previous scan, progression status is '{comparison.progression_status}' "
            f"with {comparison.percentage_change:.2f}% change."
        )
    content.append(Paragraph(summary_text, styles["BodyText"]))

    content.append(Spacer(1, 0.2 * cm))
    content.append(Paragraph("Technical Details", styles["Heading3"]))
    runtime_mode = "Demo"
    if "-trained" in scan.model_version.lower():
        runtime_mode = "Trained"
    tech_rows = [
        ("Field", "Value"),
        ("Classification Model", scan.model_version),
        ("Segmentation Model", scan.model_version),
        ("Runtime Mode", runtime_mode),
        ("Preprocessing", "Normalization + Denoise + Skull-strip approximation + Resize"),
        ("Metrics Status", "Demo/placeholder metrics shown unless training/evaluation scripts were executed"),
        ("Generated Timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
    ]
    content.append(_kv_table(tech_rows))

    content.append(Spacer(1, 0.25 * cm))
    content.append(
        Paragraph(
            "Recommendation: Please consult a certified radiologist/doctor for clinical confirmation.",
            styles["BodyText"],
        )
    )

    doc.build(content, onFirstPage=_NumberedCanvasMixin(), onLaterPages=_NumberedCanvasMixin())
    return out_path.as_posix()
