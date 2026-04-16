from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call FastAPI /predict endpoint and save predictions.csv")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--csv-path", type=str, required=True, help="Path to test csv (heart_test.csv)")
    parser.add_argument("--output", type=str, required=True, help="Where to save predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    url = f"http://{args.host}:{args.port}/predict"
    payload = {"csv_path": str(csv_path)}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    predictions = data["predictions"]

    import pandas as pd

    pred_df = pd.DataFrame(predictions, columns=["id", "prediction"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=True)

    print(f"Saved predictions via API: {out_path} (rows={len(pred_df)})")


if __name__ == "__main__":
    main()

