from dataclasses import dataclass


@dataclass(frozen=True)
class HeartConfig:
    # Columns in the dataset
    target_col: str = "Heart Attack Risk (Binary)"
    id_col: str = "id"
    gender_col: str = "Gender"

    # CSV usually contains an extra unnamed index column.
    unnamed_index_col_candidates: tuple[str, ...] = ("Unnamed: 0", "")

    # CatBoost training hyperparameters (good baseline for this dataset size).
    catboost_iterations: int = 800
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_random_seed: int = 42

    # Used for early stopping.
    early_stopping_rounds: int = 50

    # How we decide the final class from probabilities.
    # We will search the best threshold on the validation split.
    threshold_search_grid: tuple[float, ...] = tuple(
        [round(x / 100, 2) for x in range(30, 81, 1)]
    )

