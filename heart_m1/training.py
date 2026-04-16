from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import HeartConfig
from .data import HeartCSVData
from .model import (
    CatBoostHeartRiskModel,
    HeartModelMeta,
    HeartModelRepository,
)


class HeartRiskTrainer:
    def __init__(self, cfg: HeartConfig, model_dir: Path):
        self.cfg = cfg
        self.model_dir = model_dir
        self.data_builder = HeartCSVData(cfg)
        self.repo = HeartModelRepository(model_dir)

    def train_and_save(self, train_csv_path: Path) -> dict[str, float]:
        train_df = pd.read_csv(train_csv_path)
        X, y, ids, feature_spec = self.data_builder.prepare_train(train_df)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.cfg.catboost_random_seed, stratify=y
        )

        model = CatBoostHeartRiskModel()
        model.fit(
            X_train,
            y_train,
            feature_spec.catboost_categorical_feature_indices,
            cfg=self.cfg,
            X_val=X_val,
            y_val=y_val,
        )

        val_proba = model.predict_proba(X_val)
        val_auc = roc_auc_score(y_val, val_proba)

        # Choose threshold that maximizes macro-F1 on validation.
        best_threshold = 0.5
        best_f1 = -1.0
        for thr in self.cfg.threshold_search_grid:
            val_pred = [1 if p >= thr else 0 for p in val_proba]
            f1 = f1_score(y_val, val_pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr

        meta = HeartModelMeta(
            config=asdict(self.cfg),
            feature_spec=feature_spec,
            threshold=float(best_threshold),
        )
        self.repo.save_all(model=model, meta=meta)

        return {
            "val_auc": float(val_auc),
            "val_f1_macro": float(best_f1),
            "threshold": float(best_threshold),
        }

