from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from catboost import CatBoostClassifier

from .config import HeartConfig
from .data import HeartFeatureSpec


@dataclass(frozen=True)
class HeartModelMeta:
    config: dict[str, Any]
    feature_spec: HeartFeatureSpec
    threshold: float


class CatBoostHeartRiskModel:
    """
    Minimal wrapper around CatBoost for binary classification.
    """

    def __init__(self, model: Optional[CatBoostClassifier] = None):
        self.model = model or CatBoostClassifier()
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_feature_indices: list[int],
        *,
        cfg: HeartConfig,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self.model = CatBoostClassifier(
            iterations=cfg.catboost_iterations,
            learning_rate=cfg.catboost_learning_rate,
            depth=cfg.catboost_depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=cfg.catboost_random_seed,
            verbose=200,
        )

        fit_kwargs: dict[str, Any] = {"cat_features": categorical_feature_indices}
        if X_val is not None and y_val is not None:
            fit_kwargs.update(
                {
                    "eval_set": [(X_val, y_val)],
                    "use_best_model": True,
                    "early_stopping_rounds": cfg.early_stopping_rounds,
                }
            )

        self.model.fit(X, y, **fit_kwargs)
        self._is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> list[float]:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        proba = self.model.predict_proba(X)
        # For binary classification returns Nx2 -> take class 1 probability.
        return proba[:, 1].tolist()

    def predict_labels(self, X: pd.DataFrame, *, threshold: float) -> list[int]:
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))

    @staticmethod
    def load(model_path: Path) -> "CatBoostHeartRiskModel":
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        wrapper = CatBoostHeartRiskModel(model=model)
        wrapper._is_fitted = True
        return wrapper


class HeartModelRepository:
    """
    Saves/loads model + meta so that prediction code and FastAPI share one source of truth.
    """

    def __init__(self, model_dir: Path, *, model_filename: str = "heart_catboost.cbm"):
        self.model_dir = model_dir
        self.model_filename = model_filename

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def meta_path(self) -> Path:
        return self.model_dir / "meta.json"

    def save_all(self, model: CatBoostHeartRiskModel, meta: HeartModelMeta) -> None:
        model.save(self.model_path)

        # HeartFeatureSpec is not JSON serializable by default -> convert manually.
        meta_json = {
            "config": meta.config,
            "feature_spec": asdict(meta.feature_spec),
            "threshold": meta.threshold,
        }
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta_json, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_all(self) -> tuple[CatBoostHeartRiskModel, HeartModelMeta]:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")

        model = CatBoostHeartRiskModel.load(self.model_path)
        meta_raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        feature_spec_raw = meta_raw["feature_spec"]
        feature_spec = HeartFeatureSpec(**feature_spec_raw)
        meta = HeartModelMeta(
            config=meta_raw["config"],
            feature_spec=feature_spec,
            threshold=float(meta_raw["threshold"]),
        )
        return model, meta

