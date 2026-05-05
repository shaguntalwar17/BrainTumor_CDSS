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
    content.append(Spacer(1, 0.25 * cm))

    uncertainty = confidence_warning(scan.confidence_score)
    volume_text = "Area-based approximation"
    if scan.tumor_volume is not None:
        volume_text = f"{scan.tumor_volume:.2f} mm^3"

    content.append(Paragraph("1. Patient Information", styles["Heading3"]))
    patient_rows = [
        ("Field", "Value"),
        ("Patient Name", patient.name),
        ("Patient ID", patient.patient_id),
        ("Age", str(patient.age)),
        ("Gender", patient.gender),
        ("Scan Date", str(scan.scan_date)),
        ("Report Generated Date", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
    ]
    content.append(_kv_table(patient_rows))
    content.append(Spacer(1, 0.2 * cm))

    if scan.image_path and Path(scan.image_path).exists():
        content.append(Paragraph("2. Uploaded MRI Preview", styles["Heading3"]))
        content.append(Image(scan.image_path, width=7.0 * cm, height=7.0 * cm))
        content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("3. AI Tumor Detection Result", styles["Heading3"]))
    detection_rows = [
        ("Field", "Value"),
        ("Tumor Detected", "Yes" if scan.tumor_detected else "No"),
        ("Confidence Score", f"{scan.confidence_score:.3f}"),
        ("Uncertainty Score", f"{scan.uncertainty_score:.3f}" if scan.uncertainty_score is not None else "N/A"),
        ("Uncertainty Std", f"{scan.uncertainty_std:.4f}" if scan.uncertainty_std is not None else "N/A"),
        ("Stage Estimate", scan.stage_label or "Unavailable"),
        ("Stage Confidence", f"{scan.stage_confidence:.3f}" if scan.stage_confidence is not None else "N/A"),
        ("Stage Method", scan.stage_method or "N/A"),
    ]
    content.append(_kv_table(detection_rows))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("4. Tumor Classification Result", styles["Heading3"]))
    if class_probabilities:
        prob_rows = [("Class", "Probability (%)")]
        for class_name, prob in class_probabilities:
            prob_rows.append((class_name, f"{prob * 100:.2f}"))
        content.append(_kv_table(prob_rows))
    else:
        content.append(Paragraph("Class probability table unavailable for this scan context.", styles["BodyText"]))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("5. Tumor Segmentation Result", styles["Heading3"]))
    content.append(Paragraph("Binary tumor mask and overlay are generated for educational/research visualization.", styles["BodyText"]))
    if scan.mask_path and Path(scan.mask_path).exists():
        content.append(Image(scan.mask_path, width=6.5 * cm, height=6.5 * cm))
    if scan.overlay_path and Path(scan.overlay_path).exists():
        content.append(Image(scan.overlay_path, width=6.5 * cm, height=6.5 * cm))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("6. Tumor Area/Volume Estimate", styles["Heading3"]))
    area_volume_rows = [
        ("Field", "Value"),
        ("Tumor Area (pixels)", f"{scan.tumor_area:.2f}"),
        ("Tumor Volume Estimate", volume_text),
        ("2D Approximation Notice", "Yes" if scan.tumor_volume is None else "No"),
    ]
    content.append(_kv_table(area_volume_rows))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("7. Risk Category", styles["Heading3"]))
    risk_rows = [
        ("Field", "Value"),
        ("Risk Category", scan.risk_category),
        ("Uncertainty Warning", uncertainty or "None"),
    ]
    content.append(_kv_table(risk_rows))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("8. Grad-CAM Explainability Heatmap", styles["Heading3"]))
    if scan.gradcam_path and Path(scan.gradcam_path).exists():
        content.append(Image(scan.gradcam_path, width=7.0 * cm, height=7.0 * cm))
    explainability_rows = [
        ("Field", "Value"),
        ("Explainability Method", scan.xai_method or settings.xai_method),
        ("Consistency Score", f"{(scan.explainability_consistency_score or 0):.3f}"),
    ]
    content.append(_kv_table(explainability_rows))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("9. Longitudinal Comparison with Previous MRI", styles["Heading3"]))
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
        content.append(Paragraph("No previous scan available for longitudinal comparison.", styles["BodyText"]))
    content.append(Spacer(1, 0.2 * cm))

    if progression_chart_path and Path(progression_chart_path).exists():
        content.append(Paragraph("10. Tumor Progression Graph", styles["Heading3"]))
        content.append(Image(progression_chart_path, width=14.5 * cm, height=6 * cm))
        content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("11. AI Interpretation Summary", styles["Heading3"]))
    summary_text = (
        f"Stored model outputs indicate tumor detection = {scan.tumor_detected}, "
        f"tumor type = {scan.tumor_type or 'Not detected'}, confidence = {scan.confidence_score:.3f}, "
        f"risk = {scan.risk_category}, area = {scan.tumor_area:.2f} pixels, and "
        f"volume = {volume_text}."
    )
    if comparison:
        summary_text += (
            f" Previous vs current scan comparison shows status '{comparison.progression_status}' "
            f"with {comparison.percentage_change:.2f}% change."
        )
    content.append(Paragraph(summary_text, styles["BodyText"]))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("12. Model Confidence and Limitations", styles["Heading3"]))
    limitations_text = (
        "This is an AI-assisted academic prototype. Output reliability may decrease for low-confidence predictions. "
        "Stage outputs are for research/educational use only and may come from proxy burden rules when validated stage labels/models are unavailable. "
        "For 2D scans, tumor trend is area-based approximation and not true clinical volumetry."
    )
    content.append(Paragraph(limitations_text, styles["BodyText"]))
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("13. Recommendation", styles["Heading3"]))
    content.append(
        Paragraph(
            "Please consult a certified radiologist/doctor for clinical confirmation.",
            styles["BodyText"],
        )
    )
    content.append(Spacer(1, 0.2 * cm))

    content.append(Paragraph("14. Technical Appendix", styles["Heading3"]))
    runtime_mode = "Demo"
    if "-trained" in scan.model_version.lower():
        runtime_mode = "Trained"
    tech_rows = [
        ("Field", "Value"),
        ("Model Name/Version", scan.model_version),
        ("Runtime Mode", runtime_mode),
        ("Explainability Method", scan.xai_method or settings.xai_method),
        ("Preprocessing Steps", "Intensity normalization + denoise + skull-strip approximation + resize"),
        ("Metrics Note", "Actual metrics depend on executed training/evaluation runs."),
        ("Generated Timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
    ]
    content.append(_kv_table(tech_rows))

    doc.build(content, onFirstPage=_NumberedCanvasMixin(), onLaterPages=_NumberedCanvasMixin())
    return out_path.as_posix()


def generate_comparison_report(
    patient: Patient,
    previous_scan: Scan,
    current_scan: Scan,
    comparison_summary: str,
    progression_status: str,
    absolute_change: float,
    percentage_change: float,
    progression_chart_path: str | None,
    growth_map_path: str | None,
) -> str:
    ensure_dir(settings.report_dir)
    out_path = Path(settings.report_dir) / f"comparison_{previous_scan.id}_to_{current_scan.id}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_cmp",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=colors.HexColor("#102A43"),
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

    content = [
        Paragraph("AI-Assisted Brain MRI Longitudinal Comparison Report", title_style),
        Paragraph(f"<b>Disclaimer:</b> {REPORT_DISCLAIMER}", styles["BodyText"]),
        Paragraph(f"<b>Stage Note:</b> {STAGE_NOTE}", styles["BodyText"]),
        Spacer(1, 0.25 * cm),
        Paragraph("Patient Information", styles["Heading3"]),
        _kv_table(
            [
                ("Field", "Value"),
                ("Patient Name", patient.name),
                ("Patient ID", patient.patient_id),
                ("Age", str(patient.age)),
                ("Gender", patient.gender),
                ("Generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
            ]
        ),
        Spacer(1, 0.2 * cm),
        Paragraph("Comparison Summary", styles["Heading3"]),
        _kv_table(
            [
                ("Field", "Value"),
                ("Previous Scan ID / Date", f"{previous_scan.id} / {previous_scan.scan_date}"),
                ("Current Scan ID / Date", f"{current_scan.id} / {current_scan.scan_date}"),
                ("Previous Tumor Type", previous_scan.tumor_type or "N/A"),
                ("Current Tumor Type", current_scan.tumor_type or "N/A"),
                ("Previous Stage", previous_scan.stage_label or "Unavailable"),
                ("Current Stage", current_scan.stage_label or "Unavailable"),
                ("Absolute Change", f"{absolute_change:.2f}"),
                ("Percentage Change", f"{percentage_change:.2f}%"),
                ("Progression Status", progression_status),
            ]
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(comparison_summary, styles["BodyText"]),
        Spacer(1, 0.2 * cm),
    ]

    if progression_chart_path and Path(progression_chart_path).exists():
        content.append(Paragraph("Tumor Progression Graph", styles["Heading3"]))
        content.append(Image(progression_chart_path, width=14.5 * cm, height=5.8 * cm))
        content.append(Spacer(1, 0.2 * cm))

    if growth_map_path and Path(growth_map_path).exists():
        content.append(Paragraph("Tumor Growth/Reduction Map", styles["Heading3"]))
        content.append(Image(growth_map_path, width=14.5 * cm, height=5.8 * cm))
        content.append(Spacer(1, 0.2 * cm))

    content.append(
        Paragraph(
            "Recommendation: Expert radiology review is mandatory before any clinical interpretation.",
            styles["BodyText"],
        )
    )

    doc.build(content, onFirstPage=_NumberedCanvasMixin(), onLaterPages=_NumberedCanvasMixin())
    return out_path.as_posix()
