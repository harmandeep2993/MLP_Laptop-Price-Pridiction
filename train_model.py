"""
train_model.py
--------------
Trains the Laptop Price Prediction model from the cleaned dataset and saves
the pipeline (pipe.pkl) and reference dataframe (df.pkl) to the models/ directory.

Run this script whenever you need to regenerate the model artifacts:
    python train_model.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def load_data(path: str = "data/cleaned_Data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    """Build and fit a scikit-learn Pipeline on the full cleaned dataset."""
    X = df.drop("Price", axis=1)
    y = np.log(df["Price"])

    # Categorical feature indices (Company, OS, Processor, TypeName, VC)
    cat_cols = [0, 1, 2, 3, 4]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Quick evaluation
    from sklearn.metrics import r2_score, mean_absolute_error

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained  |  R² = {r2:.4f}  |  MAE (log scale) = {mae:.4f}")

    return pipeline


def save_artifacts(pipeline: Pipeline, df: pd.DataFrame) -> None:
    with open("models/pipe.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    # Save only the feature columns (drop Price) for the Streamlit UI drop-downs
    df_ref = df.drop("Price", axis=1)
    with open("models/df.pkl", "wb") as f:
        pickle.dump(df_ref, f)
    print("Artifacts saved to models/pipe.pkl and models/df.pkl")


if __name__ == "__main__":
    df = load_data()
    pipeline = build_pipeline(df)
    save_artifacts(pipeline, df)
