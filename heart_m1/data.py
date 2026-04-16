from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .config import HeartConfig


@dataclass(frozen=True)
class HeartFeatureSpec:
    feature_columns: list[str]
    categorical_columns: list[str]
    catboost_categorical_feature_indices: list[int]


class HeartCSVData:
    """
    Loading + preprocessing based on the training feature specification.
    """

    def __init__(self, config: HeartConfig):
        self.config = config

    def _drop_unnamed_index_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for candidate in self.config.unnamed_index_col_candidates:
            if candidate in df.columns:
                df = df.drop(columns=[candidate])
        return df

    def _ensure_gender_is_categorical_str(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.gender_col not in df.columns:
            return df
        df = df.copy()
        # Gender can contain both strings and numeric codes in this dataset.
        df[self.config.gender_col] = df[self.config.gender_col].fillna("NA").astype(str)
        df[self.config.gender_col] = df[self.config.gender_col].replace({"nan": "NA"})
        return df

    def build_feature_spec(self, train_df: pd.DataFrame) -> HeartFeatureSpec:
        feature_cols = [
            c
            for c in train_df.columns
            if c not in {self.config.target_col, self.config.id_col}
            and c not in self.config.unnamed_index_col_candidates
        ]

        categorical_cols = []
        if self.config.gender_col in feature_cols:
            categorical_cols.append(self.config.gender_col)

        catboost_cat_indices: list[int] = []
        for col in categorical_cols:
            catboost_cat_indices.append(feature_cols.index(col))

        return HeartFeatureSpec(
            feature_columns=feature_cols,
            categorical_columns=categorical_cols,
            catboost_categorical_feature_indices=catboost_cat_indices,
        )

    def prepare_train(
        self, train_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, HeartFeatureSpec]:
        df = self._drop_unnamed_index_columns(train_df)
        df = self._ensure_gender_is_categorical_str(df)

        if self.config.target_col not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")
        if self.config.id_col not in df.columns:
            raise ValueError(f"Missing id column: {self.config.id_col}")

        y = df[self.config.target_col].astype(float).astype(int)
        ids = df[self.config.id_col]

        feature_spec = self.build_feature_spec(df)
        X = df[feature_spec.feature_columns]
        return X, y, ids, feature_spec

    def prepare_test(self, test_df: pd.DataFrame, feature_spec: HeartFeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
        df = self._drop_unnamed_index_columns(test_df)
        df = self._ensure_gender_is_categorical_str(df)

        if self.config.id_col not in df.columns:
            raise ValueError(f"Missing id column: {self.config.id_col}")

        ids = df[self.config.id_col]

        # Ensure column order matches what model was trained on.
        missing_cols = [c for c in feature_spec.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in test: {missing_cols}")

        X = df[feature_spec.feature_columns]
        return X, ids

