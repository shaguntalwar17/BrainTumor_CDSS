from __future__ import annotations

import argparse

from ml.segmentation.train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", type=str, default="ml/configs/segmentation_config.yaml")
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--masks", type=str, default=None)
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-masks", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--best-model-name", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args)
