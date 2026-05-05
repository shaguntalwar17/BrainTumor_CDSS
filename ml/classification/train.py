from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass
class EpochStats:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    skipped_batches: int = 0


def canonical_model_name(model_name: str) -> str:
    key = model_name.lower().strip().replace(" ", "").replace("-", "_").replace("/", "_")
    aliases = {
        "resnext101_32x8d": "resnext101_32x8d",
        "resnet50": "resnet50",
        "resnet18": "resnet18",
        "densenet121": "densenet121",
        "efficientnet_b3": "efficientnet_b3",
        "efficientnetv2_s": "efficientnet_v2_s",
        "efficientnet_v2_s": "efficientnet_v2_s",
        "convnext_tiny": "convnext_tiny",
        "vit_b_16": "vit_b_16",
        "vit_b16": "vit_b_16",
        "visiontransformer": "vit_b_16",
    }
    return aliases.get(key, key)


def _display_name(model_name: str) -> str:
    model_name = canonical_model_name(model_name)
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
    return mapping.get(model_name.lower(), model_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_tf, val_tf


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    model_name = canonical_model_name(model_name)
    weights = "DEFAULT"

    if model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return model

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return model

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return model

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return model

    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        return model

    if model_name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Linear(in_features, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


class FocalCrossEntropy(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.label_smoothing = float(max(0.0, label_smoothing))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_criterion(cfg: dict, class_weights: np.ndarray, device: torch.device) -> nn.Module:
    training_cfg = cfg.get("training", {})
    use_class_weighted_loss = bool(training_cfg.get("use_class_weighted_loss", True))
    weight_tensor = None
    if use_class_weighted_loss:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_name = str(training_cfg.get("loss", "cross_entropy")).strip().lower()

    if loss_name == "focal":
        return FocalCrossEntropy(
            alpha=weight_tensor,
            gamma=float(training_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        )

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
    )


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    model_name = canonical_model_name(model_name)
    for p in model.parameters():
        p.requires_grad = False

    if model_name in {"resnext101_32x8d", "resnet50", "resnet18"}:
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name in {"efficientnet_b3", "convnext_tiny", "efficientnet_v2_s"}:
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name == "vit_b_16":
        for p in model.heads.parameters():
            p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, model_name: str) -> None:
    model_name = canonical_model_name(model_name)
    if model_name in {"resnext101_32x8d", "resnet50", "resnet18"}:
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.layer3.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
        for p in model.features.denseblock3.parameters():
            p.requires_grad = True
    elif model_name in {"efficientnet_b3", "convnext_tiny", "efficientnet_v2_s"}:
        for p in list(model.features.parameters())[-40:]:
            p.requires_grad = True
    elif model_name == "vit_b_16":
        for p in model.encoder.layers[-2:].parameters():
            p.requires_grad = True


def make_loaders(cfg: dict):
    train_tf, val_tf = make_transforms(cfg["data"]["image_size"])

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=train_tf)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=val_tf)

    targets = [y for _, y in train_ds.samples]
    class_counts = np.bincount(targets)
    sampler_power = float(cfg.get("training", {}).get("sampler_power", 1.0))
    class_weights = np.power(1.0 / np.maximum(class_counts, 1), sampler_power)
    class_weights = class_weights / np.maximum(class_weights.sum(), 1e-8) * len(class_weights)
    sample_weights = [float(class_weights[t]) for t in targets]

    use_weighted_sampler = bool(cfg.get("training", {}).get("use_weighted_sampler", True))
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            sampler=sampler,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes, class_weights


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    train: bool,
    grad_clip_norm: float | None = None,
    skip_nan_batches: bool = True,
) -> EpochStats:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    y_true = []
    y_pred = []
    skipped_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                if train and skip_nan_batches:
                    skipped_batches += 1
                    continue
                raise RuntimeError("Encountered non-finite loss during epoch run.")

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip_norm and grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    if not y_true:
        return EpochStats(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, skipped_batches=skipped_batches)

    return EpochStats(
        loss=float(np.mean(losses) if losses else 0.0),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        skipped_batches=skipped_batches,
    )


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.data:
        cfg["data"]["train_dir"] = str(Path(args.data) / "train")
        cfg["data"]["val_dir"] = str(Path(args.data) / "test")
    if args.train_dir:
        cfg["data"]["train_dir"] = args.train_dir
    if args.val_dir:
        cfg["data"]["val_dir"] = args.val_dir
    if args.epochs is not None:
        total = max(1, int(args.epochs))
        cfg["training"]["epochs_head"] = max(1, total // 3)
        cfg["training"]["epochs_finetune"] = max(1, total - cfg["training"]["epochs_head"])
    if args.epochs_head is not None:
        cfg["training"]["epochs_head"] = int(args.epochs_head)
    if args.epochs_finetune is not None:
        cfg["training"]["epochs_finetune"] = int(args.epochs_finetune)
    if args.output:
        output_path = Path(args.output)
        cfg["artifacts"]["output_dir"] = str(output_path.parent.as_posix())
        cfg["artifacts"]["best_model_name"] = output_path.name
    if args.output_dir:
        cfg["artifacts"]["output_dir"] = args.output_dir
    if args.best_model_name:
        cfg["artifacts"]["best_model_name"] = args.best_model_name
    return cfg


def _collect_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
    return y_true, y_pred


def _save_training_curves(history: list[dict], out_dir: Path) -> None:
    epochs = list(range(1, len(history) + 1))
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_f1 = [h["train"]["f1"] for h in history]
    val_f1 = [h["val"]["f1"] for h in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1, label="Train F1")
    plt.plot(epochs, val_f1, label="Val F1")
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close()


def main(config_path: str, args: argparse.Namespace | None = None):
    cfg = load_config(config_path)
    if args is not None:
        cfg = _apply_overrides(cfg, args)
    set_seed(cfg["seed"])

    out_dir = Path(cfg["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes, class_weights = make_loaders(cfg)
    class_distribution = {
        cls_name: int(sum(1 for _, label in train_loader.dataset.samples if label == idx))
        for idx, cls_name in enumerate(classes)
    }
    (out_dir / "class_distribution.json").write_text(json.dumps(class_distribution, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["training"]["model_name"].lower()
    model_display_name = _display_name(model_name)
    model = build_model(model_name, num_classes=len(classes), dropout=cfg["training"]["dropout"]).to(device)

    print(
        json.dumps(
            {
                "event": "training_start",
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "model_name": model_name,
                "num_classes": len(classes),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            }
        ),
        flush=True,
    )

    freeze_backbone(model, model_name)

    criterion = build_criterion(cfg, class_weights, device)
    grad_clip_norm = float(cfg["training"].get("gradient_clip_norm", 0.0))
    skip_nan_batches = bool(cfg["training"].get("skip_nan_batches", True))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg["training"]["lr_head"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["training"]["epochs_head"]))

    scaler = torch.amp.GradScaler("cuda") if (cfg["training"]["use_amp"] and device.type == "cuda") else None

    history = []
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    start_time = time.perf_counter()

    for epoch in range(cfg["training"]["epochs_head"]):
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            grad_clip_norm=grad_clip_norm,
            skip_nan_batches=skip_nan_batches,
        )
        va = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
            grad_clip_norm=None,
            skip_nan_batches=False,
        )
        scheduler.step()

        history.append({"phase": "head", "epoch": epoch + 1, "train": tr.__dict__, "val": va.__dict__})

        if va.f1 > best_f1:
            best_f1 = va.f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {
                    "model_name": model_display_name,
                    "classes": classes,
                    "state_dict": model.state_dict(),
                    "class_counts": class_distribution,
                    "training_config": cfg.get("training", {}),
                },
                out_dir / cfg["artifacts"]["best_model_name"],
            )
        else:
            bad_epochs += 1

        print(
            json.dumps(
                {
                    "phase": "head",
                    "epoch": epoch + 1,
                    "train_loss": tr.loss,
                    "train_f1": tr.f1,
                    "val_loss": va.loss,
                    "val_f1": va.f1,
                    "train_skipped_batches": tr.skipped_batches,
                    "best_val_f1": best_f1,
                    "bad_epochs": bad_epochs,
                }
            ),
            flush=True,
        )

        if bad_epochs >= cfg["training"]["early_stopping_patience"]:
            break

    unfreeze_last_blocks(model, model_name)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["training"]["lr_finetune"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["training"]["epochs_finetune"]))

    for epoch in range(cfg["training"]["epochs_finetune"]):
        tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            grad_clip_norm=grad_clip_norm,
            skip_nan_batches=skip_nan_batches,
        )
        va = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=False,
            grad_clip_norm=None,
            skip_nan_batches=False,
        )
        scheduler.step()

        history.append({"phase": "finetune", "epoch": epoch + 1, "train": tr.__dict__, "val": va.__dict__})

        if va.f1 > best_f1:
            best_f1 = va.f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {
                    "model_name": model_display_name,
                    "classes": classes,
                    "state_dict": model.state_dict(),
                    "class_counts": class_distribution,
                    "training_config": cfg.get("training", {}),
                },
                out_dir / cfg["artifacts"]["best_model_name"],
            )
        else:
            bad_epochs += 1

        print(
            json.dumps(
                {
                    "phase": "finetune",
                    "epoch": epoch + 1,
                    "train_loss": tr.loss,
                    "train_f1": tr.f1,
                    "val_loss": va.loss,
                    "val_f1": va.f1,
                    "train_skipped_batches": tr.skipped_batches,
                    "best_val_f1": best_f1,
                    "bad_epochs": bad_epochs,
                }
            ),
            flush=True,
        )

        if bad_epochs >= cfg["training"]["early_stopping_patience"]:
            break

    elapsed = time.perf_counter() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    best_path = out_dir / cfg["artifacts"]["best_model_name"]
    torch.save(
        {
            "model_name": model_display_name,
            "classes": classes,
            "state_dict": model.state_dict(),
            "class_counts": class_distribution,
            "training_config": cfg.get("training", {}),
        },
        best_path,
    )

    val_final = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

    x, _ = next(iter(val_loader))
    x = x[:1].to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(x)
        inference_time = time.perf_counter() - t0

    model_size_mb = best_path.stat().st_size / (1024 * 1024)

    metrics_payload = {
        "model_name": model_display_name,
        "task_type": "classification",
        "accuracy": val_final.accuracy,
        "precision": val_final.precision,
        "recall": val_final.recall,
        "f1_score": val_final.f1,
        "auc": None,
        "training_time": elapsed,
        "inference_time": inference_time,
        "model_size": model_size_mb,
        "best_use_case": "Mentor-recommended backbone tuned with transfer learning and macro-F1 priority.",
        "loss_name": str(cfg["training"].get("loss", "cross_entropy")),
        "sampler_power": float(cfg["training"].get("sampler_power", 1.0)),
        "note": "Actual metrics generated from current run.",
    }

    y_true, y_pred = _collect_predictions(model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    _save_training_curves(history, out_dir)

    (out_dir / cfg["artifacts"]["metrics_json"]).write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tumor classification models")
    parser.add_argument("--config", type=str, default="ml/configs/classification_config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Root with train/ and test/ folders")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs split across head/finetune phases")
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Checkpoint file path, e.g. models/classification/resnext101_best.pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--best-model-name", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args)
