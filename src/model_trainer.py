"""Model training, evaluation, and artifact export utilities."""

import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_COLUMN_INDICES,
    CLEANED_DATA_PATH,
    DATAFRAME_PATH,
    N_ESTIMATORS,
    PIPELINE_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_pipeline() -> Pipeline:
    """Build a scikit-learn pipeline with one-hot encoding and Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "OneHot",
                OneHotEncoder(sparse_output=False, drop="first"),
                CATEGORICAL_COLUMN_INDICES,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data, fit a pipeline, and return the fitted pipeline with splits.

    Returns
    -------
    tuple
        (fitted_pipeline, X_train, X_test, y_train, y_test)
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = np.log(df[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if pipeline is None:
        pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics on the test set.

    Returns
    -------
    dict
        Dictionary containing MAE, MSE, RMSE, and R² score.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }


def export_artifacts(
    pipeline: Pipeline,
    df: pd.DataFrame,
    pipeline_path: str = PIPELINE_PATH,
    dataframe_path: str = DATAFRAME_PATH,
    csv_path: str = CLEANED_DATA_PATH,
) -> None:
    """Persist the trained pipeline, DataFrame pickle, and cleaned CSV."""
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(dataframe_path, "wb") as f:
        pickle.dump(df, f)
    df.to_csv(csv_path, index=False)
