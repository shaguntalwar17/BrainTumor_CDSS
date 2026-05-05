from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".nii", ".nii.gz", ".dcm"}
RASTER_EXTS = {".jpg", ".jpeg", ".png"}


def _is_supported(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return True
    return path.suffix.lower() in SUPPORTED_EXTS


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_patient_id(stem: str) -> str:
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits if digits else stem.split("_")[0]


def _infer_split(path: Path) -> str:
    for part in path.parts:
        key = part.lower()
        if key in {"train", "val", "valid", "validation", "test"}:
            return "val" if key in {"val", "valid", "validation"} else key
    return "unknown"


def _read_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _candidate_mask_paths(mask_root: Path, rel_path: Path) -> list[Path]:
    candidates: list[Path] = []
    suffixes = [".png", ".jpg", ".jpeg"]
    base_rel = rel_path.with_suffix("")
    for suffix in suffixes:
        candidates.append(mask_root / f"{base_rel.as_posix()}{suffix}")

    parent = rel_path.parent
    stem = rel_path.stem
    for suffix in suffixes:
        candidates.append(mask_root / parent / f"{stem}{suffix}")

    # Handle .nii.gz name edge.
    if rel_path.name.lower().endswith(".nii.gz"):
        stem_nii = rel_path.name[:-7]
        for suffix in suffixes:
            candidates.append(mask_root / rel_path.parent / f"{stem_nii}{suffix}")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def verify_dataset(
    root_dir: str,
    output_json: str = "ml/artifacts/dataset_validation_report.json",
    expected_classes: list[str] | None = None,
    masks_root: str | None = None,
) -> dict:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    expected = {name.lower() for name in expected_classes or []}
    mask_root_path = Path(masks_root) if masks_root else None
    has_mask_root = bool(mask_root_path and mask_root_path.exists())

    report: dict = {
        "root": root.as_posix(),
        "mask_root": mask_root_path.as_posix() if has_mask_root else None,
        "summary": {
            "total_files_scanned": 0,
            "total_supported_items": 0,
            "total_images": 0,
            "total_masks": 0,
            "total_classes": 0,
            "class_distribution": {},
            "split_count": {},
            "class_imbalance_ratio": None,
            "expected_classes": sorted(expected) if expected else [],
        },
        "missing_files": [],
        "corrupt_images": [],
        "wrong_labels": [],
        "duplicate_files": [],
        "duplicate_patient_ids": [],
        "inconsistent_dimensions": [],
        "mask_alignment_issues": [],
        "missing_masks": [],
        "train_test_leakage_hashes": [],
        "augmentation_note": (
            "Recommended augmentations: rotation, flipping, contrast adjustment, random crop, brightness variation, "
            "and elastic transform for segmentation."
        ),
    }

    hashes: dict[str, str] = {}
    split_hashes: dict[str, set[str]] = defaultdict(set)
    patient_to_files: dict[str, list[str]] = defaultdict(list)
    patient_to_labels: dict[str, set[str]] = defaultdict(set)
    class_counter: Counter = Counter()
    split_counter: Counter = Counter()
    dims_by_label: dict[str, set[tuple[int, int]]] = defaultdict(set)
    image_rel_paths: list[tuple[Path, Path]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        report["summary"]["total_files_scanned"] += 1
        if not _is_supported(path):
            continue

        report["summary"]["total_supported_items"] += 1
        if path.stat().st_size == 0:
            report["missing_files"].append(path.as_posix())
            continue

        split = _infer_split(path)
        split_counter[split] += 1

        label = path.parent.name
        class_counter[label] += 1
        if expected and label.lower() not in expected:
            report["wrong_labels"].append(
                {
                    "path": path.as_posix(),
                    "found_label": label,
                    "expected_labels": sorted(expected),
                }
            )

        stem = path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem
        patient_id = _infer_patient_id(stem)
        patient_to_files[patient_id].append(path.as_posix())
        patient_to_labels[patient_id].add(label)

        if path.suffix.lower() in RASTER_EXTS:
            img = _read_gray(path)
            if img is None:
                report["corrupt_images"].append({"path": path.as_posix(), "reason": "unable to decode raster image"})
            else:
                dims_by_label[label].add(tuple(img.shape))
                lower_parts = {p.lower() for p in path.parts}
                is_mask = any(token in lower_parts for token in {"mask", "masks", "label", "labels"})
                if is_mask:
                    report["summary"]["total_masks"] += 1
                else:
                    report["summary"]["total_images"] += 1
                    rel_path = path.relative_to(root)
                    image_rel_paths.append((path, rel_path))

        file_hash = _file_hash(path)
        split_hashes[split].add(file_hash)
        if file_hash in hashes:
            report["duplicate_files"].append({"original": hashes[file_hash], "duplicate": path.as_posix()})
        else:
            hashes[file_hash] = path.as_posix()

    for patient_id, files in patient_to_files.items():
        labels = sorted(patient_to_labels[patient_id])
        if len(files) > 1:
            report["duplicate_patient_ids"].append(
                {
                    "patient_id": patient_id,
                    "count": len(files),
                    "labels": labels,
                    "files": files[:8],
                }
            )

    for label, dim_set in dims_by_label.items():
        if len(dim_set) > 1:
            report["inconsistent_dimensions"].append({"label": label, "dimensions": sorted([list(v) for v in dim_set])[:20]})

    train_hashes = split_hashes.get("train", set())
    test_hashes = split_hashes.get("test", set())
    val_hashes = split_hashes.get("val", set())
    leakage = (train_hashes & test_hashes) | (train_hashes & val_hashes) | (test_hashes & val_hashes)
    if leakage:
        report["train_test_leakage_hashes"] = sorted(list(leakage))[:200]

    if has_mask_root and mask_root_path is not None:
        for image_path, rel_path in image_rel_paths:
            maybe_masks = _candidate_mask_paths(mask_root_path, rel_path)
            existing_mask = next((mask for mask in maybe_masks if mask.exists()), None)
            if existing_mask is None:
                report["missing_masks"].append(
                    {"image": image_path.as_posix(), "expected_mask_examples": [p.as_posix() for p in maybe_masks[:2]]}
                )
                continue

            image = _read_gray(image_path)
            mask = _read_gray(existing_mask)
            if image is None or mask is None:
                report["mask_alignment_issues"].append(
                    {
                        "image": image_path.as_posix(),
                        "mask": existing_mask.as_posix(),
                        "issue": "corrupt image or mask",
                    }
                )
                continue
            if image.shape != mask.shape:
                report["mask_alignment_issues"].append(
                    {
                        "image": image_path.as_posix(),
                        "mask": existing_mask.as_posix(),
                        "issue": f"dimension mismatch image={image.shape}, mask={mask.shape}",
                    }
                )

    if class_counter:
        report["summary"]["class_distribution"] = dict(class_counter)
        report["summary"]["total_classes"] = len(class_counter)
        positive_values = [v for v in class_counter.values() if v > 0]
        if positive_values:
            report["summary"]["class_imbalance_ratio"] = round(max(positive_values) / min(positive_values), 4)

    report["summary"]["split_count"] = dict(split_counter)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dataset integrity for brain MRI research pipeline.")
    parser.add_argument("--root-dir", type=str, default="data", help="Dataset root directory to scan")
    parser.add_argument(
        "--output-json",
        type=str,
        default="ml/artifacts/dataset_validation_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--expected-classes",
        type=str,
        default="",
        help="Comma-separated class names for label validation (e.g. Glioma,Meningioma,Pituitary,No_Tumor)",
    )
    parser.add_argument(
        "--masks-root",
        type=str,
        default="",
        help="Optional segmentation masks root directory for image-mask alignment checks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    expected = [token.strip() for token in args.expected_classes.split(",") if token.strip()]
    result = verify_dataset(
        root_dir=args.root_dir,
        output_json=args.output_json,
        expected_classes=expected or None,
        masks_root=args.masks_root or None,
    )
    print(json.dumps(result, indent=2))
