from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_generator import DATA_DIR, ensure_data_exists
from src.preprocessing import FEATURE_COLUMNS, prepare_features
from src.train_model import MODEL_PATH, train_and_save_model


SAMPLE_PREDICTIONS_PATH = DATA_DIR / "sample_predictions.csv"


def assign_priority(df: pd.DataFrame, threshold: int | float) -> pd.DataFrame:
    """Assign operational priority labels based on predicted fill level."""
    result = df.copy()
    if "predicted_fill_pct" not in result.columns:
        result["predicted_fill_pct"] = pd.Series(dtype=float)

    predicted = result["predicted_fill_pct"].astype(float)
    result["priority"] = np.select(
        [
            predicted >= 90,
            (predicted >= 80) & (predicted < 90),
            (predicted >= threshold) & (predicted < 80),
            predicted < threshold,
        ],
        ["Critical", "High", "Medium", "Skip"],
        default="Skip",
    )
    return result


def _load_or_train_model():
    if not Path(MODEL_PATH).exists():
        train_and_save_model()
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        train_and_save_model()
        return joblib.load(MODEL_PATH)


def predict_fill_levels(bins_df: pd.DataFrame, threshold: int | float = 75) -> pd.DataFrame:
    """Predict current fill levels and save the latest prediction snapshot."""
    ensure_data_exists()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    result = bins_df.copy()
    if result.empty:
        result["predicted_fill_pct"] = pd.Series(dtype=float)
        result["priority"] = pd.Series(dtype=str)
        result.to_csv(SAMPLE_PREDICTIONS_PATH, index=False)
        return result

    model = _load_or_train_model()
    X = prepare_features(result[FEATURE_COLUMNS], include_target=False)
    predictions = np.clip(model.predict(X), 0, 100)
    result["predicted_fill_pct"] = np.round(predictions, 2)
    result = assign_priority(result, threshold=threshold)
    result.to_csv(SAMPLE_PREDICTIONS_PATH, index=False)

    return result
