from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify predictions.csv format (and optionally run test.py).")
    parser.add_argument("--predictions", type=str, required=True, help="Path to your predictions.csv")
    parser.add_argument("--correct", type=str, default=None, help="Path to correct_answers.csv (optional)")
    parser.add_argument(
        "--test-script",
        type=str,
        default="test.py",
        help="Path to provided test.py (optional; used only with --correct)",
    )
    return parser.parse_args()


def main() -> None:
    # Make sure we can import local project if needed.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    args = parse_args()

    import pandas as pd

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")

    df = pd.read_csv(pred_path, index_col=0)
    expected_cols = ["id", "prediction"]
    if list(df.columns) != expected_cols:
        raise SystemExit(f"Wrong columns. Expected {expected_cols}, got {list(df.columns)}")

    # Basic sanity: predictions should be 0/1.
    unique_preds = sorted(df["prediction"].dropna().unique().tolist())
    if any(p not in (0, 1) for p in unique_preds):
        raise SystemExit(f"Predictions must be 0/1. Got: {unique_preds}")

    print(f"Format OK. Rows={len(df)}. Prediction values={unique_preds}")

    if args.correct:
        correct_path = Path(args.correct)
        if not correct_path.exists():
            raise SystemExit(
                f"`--correct` file not found: {correct_path}\n"
                f"Provide the real path to correct_answers.csv, or omit `--correct` to run only format checks."
            )
        test_script = Path(args.test_script)
        if not test_script.exists():
            raise SystemExit(f"test.py not found: {test_script}")

        cmd = [sys.executable, str(test_script), "--student", str(pred_path), "--correct", str(correct_path)]
        print("Running provided test.py...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

