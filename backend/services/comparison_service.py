from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from backend.models.entities import Scan
from backend.utils.config import settings
from backend.utils.pathing import ensure_dir


@dataclass
class ComparisonOutput:
    previous_scan_id: int
    current_scan_id: int
    previous_scan_date: date
    current_scan_date: date
    previous_tumor_area: float
    current_tumor_area: float
    previous_tumor_volume: float | None
    current_tumor_volume: float | None
    absolute_change: float
    percentage_change: float
    tumor_type_change: str
    confidence_difference: float
    previous_stage_label: str | None
    current_stage_label: str | None
    stage_change: str
    previous_risk_level: str
    current_risk_level: str
    risk_level_change: str
    progression_status: str
    longitudinal_tumor_progression_index: float
    summary: str


def _progression_status(prev_metric: float, curr_metric: float, pct_change: float, prev_conf: float, curr_conf: float) -> str:
    if prev_metric > 0 and curr_metric == 0:
        return "Tumor no longer detected"
    if prev_metric == 0 and curr_metric > 0:
        return "New tumor detected"
    if -5.0 <= pct_change <= 5.0:
        return "Stable"
    if 5.0 < pct_change <= 20.0:
        return "Slightly increased"
    if pct_change > 20.0:
        return "Significantly increased"
    return "Improved"


def _risk_level_change(previous_risk: str, current_risk: str) -> str:
    if previous_risk == current_risk:
        return "No change"
    return f"{previous_risk} -> {current_risk}"


def _longitudinal_index(abs_change: float, pct_change: float, curr_conf: float, consistency: float | None) -> float:
    consistency = 1.0 if consistency is None else consistency
    score = 40.0
    score += min(35.0, max(-20.0, pct_change * 0.6))
    score += min(20.0, abs_change / 200.0)
    score += (curr_conf - 0.5) * 20.0
    score += (consistency - 0.5) * 20.0
    return float(max(0.0, min(100.0, score)))


def compare_scans(previous: Scan, current: Scan) -> ComparisonOutput:
    prev_metric = previous.tumor_volume if previous.tumor_volume is not None else previous.tumor_area
    curr_metric = current.tumor_volume if current.tumor_volume is not None else current.tumor_area

    absolute_change = float(curr_metric - prev_metric)
    if prev_metric > 0:
        percentage_change = float((absolute_change / prev_metric) * 100.0)
    elif curr_metric > 0:
        percentage_change = 100.0
    else:
        percentage_change = 0.0

    progression_status = _progression_status(
        prev_metric=prev_metric,
        curr_metric=curr_metric,
        pct_change=percentage_change,
        prev_conf=previous.confidence_score,
        curr_conf=current.confidence_score,
    )
    tumor_type_change = (
        "No change"
        if (previous.tumor_type or "Unknown") == (current.tumor_type or "Unknown")
        else f"{previous.tumor_type or 'Unknown'} -> {current.tumor_type or 'Unknown'}"
    )
    confidence_diff = float(current.confidence_score - previous.confidence_score)
    previous_stage = previous.stage_label
    current_stage = current.stage_label
    if (previous_stage or "N/A") == (current_stage or "N/A"):
        stage_change = "No change"
    else:
        stage_change = f"{previous_stage or 'N/A'} -> {current_stage or 'N/A'}"
    lti = _longitudinal_index(
        abs_change=absolute_change,
        pct_change=percentage_change,
        curr_conf=current.confidence_score,
        consistency=current.explainability_consistency_score,
    )
    risk_change = _risk_level_change(previous.risk_category, current.risk_category)

    low_confidence_note = ""
    if (
        previous.confidence_score < settings.low_confidence_threshold
        or current.confidence_score < settings.low_confidence_threshold
    ):
        low_confidence_note = " Low-confidence comparison: expert review required."

    summary = (
        f"Scan {previous.id} ({previous.scan_date}) vs Scan {current.id} ({current.scan_date}): "
        f"metric change {absolute_change:.2f} ({percentage_change:.2f}%). "
        f"Progression status: {progression_status}. Risk change: {risk_change}.{low_confidence_note}"
    )

    return ComparisonOutput(
        previous_scan_id=previous.id,
        current_scan_id=current.id,
        previous_scan_date=previous.scan_date,
        current_scan_date=current.scan_date,
        previous_tumor_area=previous.tumor_area,
        current_tumor_area=current.tumor_area,
        previous_tumor_volume=previous.tumor_volume,
        current_tumor_volume=current.tumor_volume,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        tumor_type_change=tumor_type_change,
        confidence_difference=confidence_diff,
        previous_stage_label=previous_stage,
        current_stage_label=current_stage,
        stage_change=stage_change,
        previous_risk_level=previous.risk_category,
        current_risk_level=current.risk_category,
        risk_level_change=risk_change,
        progression_status=progression_status,
        longitudinal_tumor_progression_index=lti,
        summary=summary,
    )


def build_growth_chart(patient_id: str, scans: list[Scan]) -> str | None:
    if not scans:
        return None

    scans_sorted = sorted(scans, key=lambda s: s.scan_date)
    x_labels = [str(s.scan_date) for s in scans_sorted]
    y_vals = [s.tumor_volume if s.tumor_volume is not None else s.tumor_area for s in scans_sorted]

    ensure_dir(settings.chart_dir)
    out_path = Path(settings.chart_dir) / f"patient_{patient_id}_progression.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_labels, y_vals, marker="o", color="#008B8B", linewidth=2)
    plt.title("Tumor Growth Over Time (Area/Volume)")
    plt.xlabel("Scan Date")
    plt.ylabel("Tumor Metric")
    plt.xticks(rotation=30, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return out_path.as_posix()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return token or "patient"


def build_growth_change_map(
    patient_id: str,
    previous_scan_id: int,
    current_scan_id: int,
    previous_mask_path: str | None,
    current_mask_path: str | None,
) -> str | None:
    if not previous_mask_path or not current_mask_path:
        return None

    prev_path = Path(previous_mask_path)
    curr_path = Path(current_mask_path)
    if not prev_path.exists() or not curr_path.exists():
        return None

    prev = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(curr_path), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        return None

    if prev.shape != curr.shape:
        curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_NEAREST)

    prev_bin = prev > 0
    curr_bin = curr > 0

    stable = np.logical_and(prev_bin, curr_bin)
    growth = np.logical_and(curr_bin, np.logical_not(prev_bin))
    regression = np.logical_and(prev_bin, np.logical_not(curr_bin))

    canvas = np.zeros((prev.shape[0], prev.shape[1], 3), dtype=np.uint8)
    canvas[stable] = (0, 220, 220)      # Stable overlap: cyan
    canvas[growth] = (20, 20, 240)      # New growth: red
    canvas[regression] = (20, 140, 240)  # Reduced region: orange

    legend_strip = np.zeros((42, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend_strip, "Growth map", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        legend_strip,
        "Red=New/Increased | Cyan=Stable | Orange=Reduced",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    output_img = np.concatenate([legend_strip, canvas], axis=0)
    ensure_dir(settings.chart_dir)
    out_path = (
        Path(settings.chart_dir)
        / f"patient_{_safe_token(patient_id)}_growth_map_{previous_scan_id}_to_{current_scan_id}.png"
    )
    cv2.imwrite(str(out_path), output_img)
    return out_path.as_posix()
