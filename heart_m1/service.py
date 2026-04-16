from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import HeartConfig
from .model import HeartModelRepository
from .predicting import HeartRiskPredictor
from .training import HeartRiskTrainer


class HeartRiskService:
    """
    High-level facade used by scripts and FastAPI.
    """

    def __init__(self, cfg: HeartConfig, model_dir: Path):
        self.cfg = cfg
        self.model_dir = model_dir
        self.repo = HeartModelRepository(model_dir)
        self.trainer = HeartRiskTrainer(cfg=cfg, model_dir=model_dir)

    def train(self, train_csv_path: Path) -> dict[str, Any]:
        return self.trainer.train_and_save(train_csv_path=train_csv_path)

    def predict_csv(self, test_csv_path: Path, output_csv_path: Path) -> pd.DataFrame:
        predictor = HeartRiskPredictor(cfg=self.cfg, model_dir=self.model_dir)
        return predictor.predict_csv(test_csv_path=test_csv_path, output_csv_path=output_csv_path)

    def predict_json(self, test_csv_path: Path) -> dict[str, Any]:
        predictor = HeartRiskPredictor(cfg=self.cfg, model_dir=self.model_dir)
        df = pd.read_csv(test_csv_path)
        return self.predict_json_from_df(df)

    def predict_json_from_df(self, df: pd.DataFrame) -> dict[str, Any]:
        predictor = HeartRiskPredictor(cfg=self.cfg, model_dir=self.model_dir)
        pred_df = predictor.predict_df(df)
        records = [{"id": int(rid), "prediction": int(rpred)} for rid, rpred in zip(pred_df["id"], pred_df["prediction"])]
        return {"predictions": records}

