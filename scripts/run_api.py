from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastAPI service (uvicorn)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    # Ensure project root is importable when executing `python scripts/run_api.py`.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    args = parse_args()
    uvicorn.run("heart_m1.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

