from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Heart Risk model (CatBoost)")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to heart_train.csv")
    parser.add_argument(
        "--model-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "models"), help="Where to store model"
    )
    return parser.parse_args()


def main() -> None:
    # Ensure project root is importable when executing `python scripts/...`.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from heart_m1.config import HeartConfig
    from heart_m1.service import HeartRiskService

    args = parse_args()
    cfg = HeartConfig()
    service = HeartRiskService(cfg=cfg, model_dir=Path(args.model_dir))
    metrics = service.train(train_csv_path=Path(args.train_csv))
    print("Training metrics:", metrics)


if __name__ == "__main__":
    main()

