"""Prediction helper used by the Streamlit application."""

import pickle

import numpy as np

from .config import DATAFRAME_PATH, PIPELINE_PATH


def load_artifacts(
    pipeline_path: str = PIPELINE_PATH,
    dataframe_path: str = DATAFRAME_PATH,
):
    """Load the trained pipeline and reference DataFrame from disk.

    Returns
    -------
    tuple
        (pipeline, dataframe)
    """
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(dataframe_path, "rb") as f:
        df = pickle.load(f)
    return pipeline, df


def predict_price(pipeline, features: np.ndarray) -> int:
    """Return the predicted price (INR) for a single feature vector.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained model pipeline.
    features : np.ndarray
        1-D array of input features matching the training schema.

    Returns
    -------
    int
        Predicted price in INR (inverse-log transformed).
    """
    query = np.array(features).reshape(1, -1)
    return int(np.exp(pipeline.predict(query)[0]))
