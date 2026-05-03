from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ml.segmentation.models import build_segmentation_model
from ml.segmentation.train import BrainSegDataset, dice_coeff, iou_score, precision_recall_specificity


def evaluate(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / cfg["artifacts"]["best_model_name"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_segmentation_model(ckpt["model_name"], in_channels=1, out_channels=1)
    model.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    allow_pseudo_masks = bool(cfg["data"].get("allow_pseudo_masks_if_missing", False))
    val_ds = BrainSegDataset(
        cfg["data"]["val_images_dir"],
        cfg["data"]["val_masks_dir"],
        cfg["data"]["image_size"],
        allow_pseudo_masks=allow_pseudo_masks,
    )
    loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    dices = []
    ious = []
    precisions = []
    recalls = []
    specs = []

    sample_saved = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)

            dices.append(dice_coeff(prob, y).item())
            ious.append(iou_score(prob, y).item())
            p, r, s = precision_recall_specificity(prob, y)
            precisions.append(p.item())
            recalls.append(r.item())
            specs.append(s.item())

            preds = (prob > 0.5).float().cpu().numpy()
            imgs = x.cpu().numpy()
            gts = y.cpu().numpy()

            for i in range(min(len(preds), 3)):
                if sample_saved >= 12:
                    break
                im = (imgs[i, 0] * 255).astype(np.uint8)
                gt = (gts[i, 0] > 0.5).astype(np.uint8) * 255
                pr = (preds[i, 0] > 0.5).astype(np.uint8) * 255

                overlay = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                overlay[pr > 0] = [0, 255, 0]

                canvas = np.concatenate([cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pr, cv2.COLOR_GRAY2BGR), overlay], axis=1)
                cv2.imwrite(str(out_dir / f"sample_pred_{sample_saved:03d}.png"), canvas)
                sample_saved += 1

    metrics = {
        "model_name": ckpt["model_name"],
        "task_type": "segmentation",
        "dice": float(np.mean(dices) if dices else 0.0),
        "iou": float(np.mean(ious) if ious else 0.0),
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "specificity": float(np.mean(specs) if specs else 0.0),
        "hausdorff95": None,
        "note": (
            "Actual metrics generated from current run."
            if not allow_pseudo_masks
            else "Metrics evaluated in pseudo-mask demo mode (ground-truth masks were unavailable)."
        )
    }

    (out_dir / "segmentation_eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 4))
    plt.bar(["Dice", "IoU", "Precision", "Recall", "Specificity"], [metrics["dice"], metrics["iou"], metrics["precision"], metrics["recall"], metrics["specificity"]], color=["#0EA5E9", "#22C55E", "#A855F7", "#F59E0B", "#14B8A6"])
    plt.ylim(0, 1)
    plt.title("Segmentation Validation Metrics")
    plt.tight_layout()
    plt.savefig(out_dir / "segmentation_metrics_bar.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--config", type=str, default="ml/configs/segmentation_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
