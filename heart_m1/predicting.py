from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import HeartConfig
from .data import HeartCSVData, HeartFeatureSpec
from .model import CatBoostHeartRiskModel, HeartModelMeta, HeartModelRepository


class HeartRiskPredictor:
    def __init__(self, cfg: HeartConfig, model_dir: Path):
        self.cfg = cfg
        self.repo = HeartModelRepository(model_dir)
        self.model, self.meta = self.repo.load_all()
        # Ensure we use feature spec produced during training.
        self.data_builder = HeartCSVData(cfg)

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X, ids = self.data_builder.prepare_test(df, self.meta.feature_spec)
        preds = self.model.predict_labels(X, threshold=self.meta.threshold)

        out = pd.DataFrame({"id": ids.values, "prediction": preds})
        return out

    def predict_csv(self, test_csv_path: Path, output_csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(test_csv_path)
        out = self.predict_df(df)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # test.py uses index_col=0; we need an "index" column in the csv.
        out.to_csv(output_csv_path, index=True)
        return out

