from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

VALID_EXTS = {".jpg", ".jpeg", ".png", ".nii", ".gz", ".dcm"}


def _file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_patient_id(stem: str) -> str:
    token = "".join(ch for ch in stem if ch.isdigit())
    return token if token else stem.split("_")[0]


def _split_name(path: Path) -> str:
    for part in path.parts:
        low = part.lower()
        if low in {"train", "val", "valid", "validation", "test"}:
            return "val" if low in {"val", "valid", "validation"} else low
    return "unknown"


def verify_dataset(root_dir: str, output_json: str = "ml/artifacts/dataset_validation_report.json") -> dict:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    report = {
        "root": root.as_posix(),
        "summary": {
            "total_images": 0,
            "total_masks": 0,
            "total_classes": 0,
            "class_distribution": {},
            "split_count": {},
            "class_imbalance_ratio": None,
        },
        "missing_files": [],
        "corrupt_images": [],
        "wrong_labels": [],
        "duplicate_files": [],
        "duplicate_patient_ids": [],
        "inconsistent_dimensions": [],
        "mask_alignment_issues": [],
        "train_test_leakage_hashes": [],
        "augmentation_note": "Recommended augmentations: rotation, flipping, contrast jitter, brightness variation, random crop, elastic transform (for segmentation).",
    }

    hashes = {}
    split_hashes = defaultdict(set)
    patient_index = defaultdict(list)
    dims = defaultdict(set)
    class_counter = Counter()
    split_counter = Counter()

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in VALID_EXTS and not path.name.lower().endswith(".nii.gz"):
            continue

        if path.stat().st_size == 0:
            report["missing_files"].append(path.as_posix())
            continue

        split = _split_name(path)
        split_counter[split] += 1

        label = path.parent.name
        class_counter[label] += 1

        stem = path.stem
        patient_id = _infer_patient_id(stem)
        patient_index[patient_id].append(path.as_posix())

        if ext in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                report["corrupt_images"].append(path.as_posix())
            else:
                dims[label].add(tuple(img.shape))
                report["summary"]["total_images"] += 1

        h = _file_hash(path)
        split_hashes[split].add(h)
        if h in hashes:
            report["duplicate_files"].append({"original": hashes[h], "duplicate": path.as_posix()})
        else:
            hashes[h] = path.as_posix()

    for pid, files in patient_index.items():
        if len(files) > 1:
            report["duplicate_patient_ids"].append({"patient_id": pid, "count": len(files), "files": files[:5]})

    for label, size_set in dims.items():
        if len(size_set) > 1:
            report["inconsistent_dimensions"].append({"label": label, "dimensions": sorted(list(size_set))[:10]})

    train_hashes = split_hashes.get("train", set())
    test_hashes = split_hashes.get("test", set())
    val_hashes = split_hashes.get("val", set())
    leakage = (train_hashes & test_hashes) | (train_hashes & val_hashes) | (test_hashes & val_hashes)
    if leakage:
        report["train_test_leakage_hashes"] = list(leakage)[:100]

    if class_counter:
        report["summary"]["class_distribution"] = dict(class_counter)
        report["summary"]["total_classes"] = len(class_counter)
        values = [v for v in class_counter.values() if v > 0]
        if values:
            report["summary"]["class_imbalance_ratio"] = round(max(values) / min(values), 4)

    report["summary"]["split_count"] = dict(split_counter)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = verify_dataset("data")
    print(json.dumps(result, indent=2))
