from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml


def _read_json_lines(log_path: Path) -> list[dict]:
    events: list[dict] = []
    if not log_path.exists():
        return events
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _build_snapshot(events: list[dict], cfg: dict, log_path: Path) -> dict:
    head_total = int(cfg.get("training", {}).get("epochs_head", 0))
    finetune_total = int(cfg.get("training", {}).get("epochs_finetune", 0))
    total_epochs = head_total + finetune_total

    epoch_events = [e for e in events if e.get("phase") in {"head", "finetune"} and "epoch" in e]
    completed = len(epoch_events)
    progress_pct = (completed / total_epochs * 100.0) if total_epochs > 0 else 0.0

    start_ts = log_path.stat().st_ctime if log_path.exists() else time.time()
    elapsed_sec = max(0.0, time.time() - start_ts)
    avg_epoch_sec = (elapsed_sec / completed) if completed > 0 else None
    remaining_epochs = max(0, total_epochs - completed)
    eta_sec = (avg_epoch_sec * remaining_epochs) if avg_epoch_sec is not None else None
    eta_finish = (
        datetime.fromtimestamp(time.time() + eta_sec).isoformat(timespec="seconds")
        if eta_sec is not None
        else None
    )

    latest = epoch_events[-1] if epoch_events else {}
    best_val_f1 = max((float(e.get("best_val_f1", -1.0)) for e in epoch_events), default=None)

    return {
        "total_epochs": total_epochs,
        "completed_epochs": completed,
        "progress_percent": round(progress_pct, 2),
        "elapsed_sec_estimate": round(elapsed_sec, 2),
        "avg_epoch_sec_estimate": round(avg_epoch_sec, 2) if avg_epoch_sec is not None else None,
        "remaining_epochs_estimate": remaining_epochs,
        "eta_sec_estimate": round(eta_sec, 2) if eta_sec is not None else None,
        "eta_finish_local_estimate": eta_finish,
        "latest_phase": latest.get("phase"),
        "latest_epoch": latest.get("epoch"),
        "latest_train_f1": latest.get("train_f1"),
        "latest_val_f1": latest.get("val_f1"),
        "latest_val_loss": latest.get("val_loss"),
        "best_val_f1_so_far": best_val_f1 if best_val_f1 is not None and best_val_f1 >= 0 else None,
    }


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {"training": {"epochs_head": 0, "epochs_finetune": 0}}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse classification training log and print ETA/progress summary.")
    parser.add_argument("--log", default="logs/classification_train.log", help="Path to JSON-lines training log.")
    parser.add_argument("--config", default="ml/configs/classification_config.yaml", help="Training config path.")
    parser.add_argument("--watch", action="store_true", help="Continuously print snapshot every interval seconds.")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch interval in seconds.")
    args = parser.parse_args()

    log_path = Path(args.log)
    cfg_path = Path(args.config)

    if args.watch:
        try:
            while True:
                events = _read_json_lines(log_path)
                cfg = _load_cfg(cfg_path)
                print(json.dumps(_build_snapshot(events, cfg, log_path), indent=2))
                print("-" * 80)
                time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            return
    else:
        events = _read_json_lines(log_path)
        cfg = _load_cfg(cfg_path)
        print(json.dumps(_build_snapshot(events, cfg, log_path), indent=2))


if __name__ == "__main__":
    main()
