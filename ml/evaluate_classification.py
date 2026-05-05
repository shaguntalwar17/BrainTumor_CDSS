from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.classification.train import build_model, canonical_model_name


def _display_name(model_name: str) -> str:
    key = canonical_model_name(model_name)
    mapping = {
        "resnext101_32x8d": "ResNeXt101_32x8d",
        "resnet50": "ResNet50",
        "resnet18": "ResNet18",
        "densenet121": "DenseNet121",
        "efficientnet_b3": "EfficientNet-B3",
        "efficientnet_v2_s": "EfficientNetV2-S",
        "convnext_tiny": "ConvNeXt-Tiny",
        "vit_b_16": "ViT-B/16",
    }
    return mapping.get(key, model_name)


def evaluate(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / cfg["artifacts"]["best_model_name"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    classes = ckpt["classes"]

    model = build_model(model_name, num_classes=len(classes), dropout=cfg["training"]["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_tf = transforms.Compose(
        [
            transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=val_tf)
    loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_prob = np.array(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_wise_report": classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0),
    }

    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
        auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        metrics["roc_auc_ovr_macro"] = float(auc)
    except Exception:
        metrics["roc_auc_ovr_macro"] = None

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=30, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    confidences = y_prob.max(axis=1)
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=20, color="#0EA5E9", alpha=0.85)
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_distribution.png", dpi=180)
    plt.close()

    (out_dir / "classification_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Also publish a backend-ready metrics row using evaluated scores (incl. AUC)
    # while preserving training-time metadata if available.
    training_metrics_path = out_dir / cfg["artifacts"]["metrics_json"]
    training_payload: dict = {}
    if training_metrics_path.exists():
        try:
            training_payload = json.loads(training_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            training_payload = {}

    backend_row = {
        "model_name": _display_name(model_name),
        "task_type": "classification",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "f1_score": metrics["f1_macro"],
        "auc": metrics.get("roc_auc_ovr_macro"),
        "training_time": training_payload.get("training_time"),
        "inference_time": training_payload.get("inference_time"),
        "model_size": training_payload.get("model_size"),
        "best_use_case": training_payload.get(
            "best_use_case",
            "Mentor-recommended backbone tuned with transfer learning and macro-F1 priority.",
        ),
        "note": "Actual evaluated metrics from classification evaluation script.",
    }
    (out_dir / "classification_metrics_evaluated.json").write_text(json.dumps(backend_row, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("--config", type=str, default="ml/configs/classification_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
