from __future__ import annotations

import argparse

from ml.evaluate_classification import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stage estimation model (research)")
    parser.add_argument("--config", type=str, default="ml/configs/stage_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
