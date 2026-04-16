from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict for heart_test.csv -> predictions.csv")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to heart_test.csv (no target column)")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to save predictions.csv (must be compatible with test.py)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory with saved model + meta.json",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from heart_m1.config import HeartConfig
    from heart_m1.service import HeartRiskService

    args = parse_args()
    cfg = HeartConfig()
    service = HeartRiskService(cfg=cfg, model_dir=Path(args.model_dir))
    out_df = service.predict_csv(
        test_csv_path=Path(args.test_csv),
        output_csv_path=Path(args.output),
    )
    print(f"Saved predictions: {args.output} (rows={len(out_df)})")


if __name__ == "__main__":
    main()

