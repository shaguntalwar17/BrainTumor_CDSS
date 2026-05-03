from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from ml.segmentation.models import build_segmentation_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SegStats:
    loss: float
    dice: float
    iou: float
    precision: float
    recall: float
    specificity: float


class BrainSegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, image_size: int = 224, allow_pseudo_masks: bool = False):
        self.image_size = image_size
        self.allow_pseudo_masks = allow_pseudo_masks
        self.images = sorted([p for p in Path(images_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.masks_dir = Path(masks_dir)

        self.items = []
        for img_path in self.images:
            mask_path = self.masks_dir / img_path.relative_to(images_dir)
            if mask_path.exists():
                self.items.append((img_path, mask_path))
            elif self.allow_pseudo_masks:
                self.items.append((img_path, None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback pseudo-mask for demo/research mode when true masks are unavailable.
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = (img.astype(np.float32) / 255.0)[None, :, :]
        mask = ((mask.astype(np.float32) / 255.0) > 0.5).astype(np.float32)[None, :, :]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = (pred_bin + target - pred_bin * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean()


def precision_recall_specificity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred_bin = (pred > 0.5).float()
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred_bin) * (1 - target)).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * target).sum(dim=(1, 2, 3))

    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()
    specificity = ((tn + eps) / (tn + fp + eps)).mean()
    return precision, recall, specificity


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        bce = self.bce(logits, target)
        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return bce + dice_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        inter = (prob * target).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return focal + dice_loss


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> SegStats:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    dices = []
    ious = []
    precisions = []
    recalls = []
    specs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
                prob = torch.sigmoid(logits)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        d = dice_coeff(prob, y)
        j = iou_score(prob, y)
        p, r, s = precision_recall_specificity(prob, y)

        losses.append(loss.item())
        dices.append(d.item())
        ious.append(j.item())
        precisions.append(p.item())
        recalls.append(r.item())
        specs.append(s.item())

    return SegStats(
        loss=float(np.mean(losses) if losses else 0.0),
        dice=float(np.mean(dices) if dices else 0.0),
        iou=float(np.mean(ious) if ious else 0.0),
        precision=float(np.mean(precisions) if precisions else 0.0),
        recall=float(np.mean(recalls) if recalls else 0.0),
        specificity=float(np.mean(specs) if specs else 0.0),
    )


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.images:
        cfg["data"]["images_dir"] = args.images
    if args.masks:
        cfg["data"]["masks_dir"] = args.masks
    if args.val_images:
        cfg["data"]["val_images_dir"] = args.val_images
    if args.val_masks:
        cfg["data"]["val_masks_dir"] = args.val_masks
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.output:
        output_path = Path(args.output)
        cfg["artifacts"]["output_dir"] = str(output_path.parent.as_posix())
        cfg["artifacts"]["best_model_name"] = output_path.name
    if args.output_dir:
        cfg["artifacts"]["output_dir"] = args.output_dir
    if args.best_model_name:
        cfg["artifacts"]["best_model_name"] = args.best_model_name
    return cfg


def _save_curves(history: list[dict], out_dir: Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_dice = [h["train"]["dice"] for h in history]
    val_dice = [h["val"]["dice"] for h in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Segmentation Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.title("Segmentation Dice Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "segmentation_curves.png", dpi=180)
    plt.close()


def main(config_path: str, args: argparse.Namespace | None = None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if args is not None:
        cfg = _apply_overrides(cfg, args)
    set_seed(cfg["seed"])

    out_dir = Path(cfg["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_pseudo_masks = bool(cfg["data"].get("allow_pseudo_masks_if_missing", False))
    train_ds = BrainSegDataset(
        cfg["data"]["images_dir"],
        cfg["data"]["masks_dir"],
        cfg["data"]["image_size"],
        allow_pseudo_masks=allow_pseudo_masks,
    )
    val_ds = BrainSegDataset(
        cfg["data"]["val_images_dir"],
        cfg["data"]["val_masks_dir"],
        cfg["data"]["image_size"],
        allow_pseudo_masks=allow_pseudo_masks,
    )

    if len(train_ds) == 0:
        raise RuntimeError("No training data found. Ensure image and mask paths are configured correctly.")
    if len(val_ds) == 0:
        raise RuntimeError("No validation data found. Ensure image and mask paths are configured correctly.")

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    (out_dir / "dataset_report.json").write_text(
        json.dumps(
            {
                "train_pairs": len(train_ds),
                "val_pairs": len(val_ds),
                "images_dir": cfg["data"]["images_dir"],
                "masks_dir": cfg["data"]["masks_dir"],
                "val_images_dir": cfg["data"]["val_images_dir"],
                "val_masks_dir": cfg["data"]["val_masks_dir"],
                "allow_pseudo_masks_if_missing": allow_pseudo_masks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(cfg["training"]["model_name"], in_channels=1, out_channels=1).to(device)

    if cfg["training"]["loss"].lower() == "dice_focal":
        criterion = DiceFocalLoss()
    else:
        criterion = DiceBCELoss()

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda") if (cfg["training"]["use_amp"] and device.type == "cuda") else None

    best_dice = -1.0
    best_state = None
    bad_epochs = 0
    history = []
    start = time.perf_counter()

    for epoch in range(cfg["training"]["epochs"]):
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

        scheduler.step(va.dice)

        history.append({"epoch": epoch + 1, "train": tr.__dict__, "val": va.__dict__})

        if va.dice > best_dice:
            best_dice = va.dice
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg["training"]["early_stopping_patience"]:
            break

    elapsed = time.perf_counter() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    best_path = out_dir / cfg["artifacts"]["best_model_name"]
    torch.save({"model_name": cfg["training"]["model_name"], "state_dict": model.state_dict()}, best_path)

    final = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

    metrics = {
        "model_name": cfg["training"]["model_name"],
        "task_type": "segmentation",
        "dice": final.dice,
        "iou": final.iou,
        "precision": final.precision,
        "recall": final.recall,
        "specificity": final.specificity,
        "hausdorff95": None,
        "training_time": elapsed,
        "inference_time": None,
        "model_size": best_path.stat().st_size / (1024 * 1024),
        "note": (
            "Actual metrics generated from current run."
            if not allow_pseudo_masks
            else "Metrics generated in pseudo-mask demo mode (ground-truth masks were unavailable)."
        ),
    }

    (out_dir / cfg["artifacts"]["metrics_json"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_curves(history, out_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models")
    parser.add_argument("--config", type=str, default="ml/configs/segmentation_config.yaml")
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--masks", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-masks", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path, e.g. models/segmentation/unet_best.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--best-model-name", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args)
