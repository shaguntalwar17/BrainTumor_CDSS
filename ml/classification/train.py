from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
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
    model_name = model_name.lower()
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

    raise ValueError(f"Unsupported model name: {model_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "resnext101_32x8d" or model_name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif model_name in {"efficientnet_b3", "convnext_tiny"}:
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, model_name: str) -> None:
    if model_name in {"resnext101_32x8d", "resnet50"}:
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.layer3.parameters():
            p.requires_grad = True
    elif model_name == "densenet121":
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
        for p in model.features.denseblock3.parameters():
            p.requires_grad = True
    elif model_name in {"efficientnet_b3", "convnext_tiny"}:
        for p in list(model.features.parameters())[-40:]:
            p.requires_grad = True


def make_loaders(cfg: dict):
    train_tf, val_tf = make_transforms(cfg["data"]["image_size"])

    train_ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=train_tf)
    val_ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=val_tf)

    targets = [y for _, y in train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        sampler=sampler,
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


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> EpochStats:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    return EpochStats(
        loss=float(np.mean(losses) if losses else 0.0),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
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

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=cfg["training"]["label_smoothing"],
    )

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
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        history.append({"phase": "head", "epoch": epoch + 1, "train": tr.__dict__, "val": va.__dict__})

        if va.f1 > best_f1:
            best_f1 = va.f1
            best_state = model.state_dict()
            bad_epochs = 0
            torch.save({"model_name": model_name, "classes": classes, "state_dict": model.state_dict()}, out_dir / cfg["artifacts"]["best_model_name"])
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
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        history.append({"phase": "finetune", "epoch": epoch + 1, "train": tr.__dict__, "val": va.__dict__})

        if va.f1 > best_f1:
            best_f1 = va.f1
            best_state = model.state_dict()
            bad_epochs = 0
            torch.save({"model_name": model_name, "classes": classes, "state_dict": model.state_dict()}, out_dir / cfg["artifacts"]["best_model_name"])
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
    torch.save({"model_name": model_name, "classes": classes, "state_dict": model.state_dict()}, best_path)

    val_final = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

    x, _ = next(iter(val_loader))
    x = x[:1].to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = model(x)
        inference_time = time.perf_counter() - t0

    model_size_mb = best_path.stat().st_size / (1024 * 1024)

    metrics_payload = {
        "model_name": model_name,
        "task_type": "classification",
        "accuracy": val_final.accuracy,
        "precision": val_final.precision,
        "recall": val_final.recall,
        "f1_score": val_final.f1,
        "auc": None,
        "training_time": elapsed,
        "inference_time": inference_time,
        "model_size": model_size_mb,
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
