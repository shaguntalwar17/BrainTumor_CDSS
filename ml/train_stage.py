from __future__ import annotations

import argparse

from ml.classification.train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stage estimation model (research)")
    parser.add_argument("--config", type=str, default="ml/configs/stage_config.yaml")
    parser.add_argument("--data", type=str, default=None, help="Root dir containing train/ and test/")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs-head", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path override")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--best-model-name", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args)
