from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.services.inference_service import analyze_scan
from backend.services.preprocessing_service import preprocess_scan
from backend.services.storage_service import save_gradcam, save_mask, save_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-scan inference (research/demo mode).")
    parser.add_argument("--image", required=True, help="Path to MRI image (.jpg/.png/.nii/.nii.gz/.dcm)")
    parser.add_argument("--scan-id", type=int, default=999999, help="Virtual scan id for saved artifacts")
    parser.add_argument("--output-json", default="ml/artifacts/inference_result.json")
    args = parser.parse_args()

    pre = preprocess_scan(args.image)
    result = analyze_scan(pre)

    mask_path = save_mask(result.mask, args.scan_id)
    gradcam_path = save_gradcam(result.gradcam, args.scan_id)
    overlay_path = save_overlay(result.overlay, args.scan_id)

    payload = {
        "tumor_detected": result.tumor_detected,
        "tumor_type": result.tumor_type,
        "confidence": result.confidence,
        "uncertainty_score": result.uncertainty_score,
        "uncertainty_std": result.uncertainty_std,
        "class_probabilities": result.class_probabilities,
        "tumor_area": result.tumor_area,
        "tumor_volume": result.tumor_volume,
        "volume_units": result.volume_units,
        "is_area_based_approximation": result.is_area_based_approximation,
        "explainability_consistency_score": result.explainability_consistency_score,
        "explainability_warning": result.explainability_warning,
        "xai_method": result.xai_method,
        "inference_time_sec": result.inference_time_sec,
        "mask_path": mask_path,
        "gradcam_path": gradcam_path,
        "overlay_path": overlay_path,
        "note": "If trained checkpoints are unavailable, this runs in research/demo heuristic inference mode.",
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
