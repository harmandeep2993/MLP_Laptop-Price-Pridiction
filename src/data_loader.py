"""Load and perform initial cleaning on the raw laptop dataset."""

import pandas as pd

from .config import RAW_DATA_PATH


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    return pd.read_csv(path)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns from the raw DataFrame.

    - Removes the ``Unnamed: 0`` index column if present.
    """
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df
