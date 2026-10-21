"""Feature engineering shared by training and serving.

The same function runs inside the saved model pipeline, so the API and the
training job always compute features identically (no training/serving skew).
"""

import numpy as np
import pandas as pd

from scoring.config import (
    CATEGORICAL_INPUTS,
    DAYS_EMPLOYED_SENTINEL,
    NUMERIC_INPUTS,
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN instead of inf when the denominator is 0 or missing."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw application fields into model features.

    Returns a new DataFrame; the input is never modified.
    """
    df = raw.copy()

    for column in NUMERIC_INPUTS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_INPUTS:
        df[column] = df[column].astype("object")

    # 365243 means "not employed": keep that fact as a flag, treat the value as missing.
    anomaly = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("float64")
    days_employed = df["DAYS_EMPLOYED"].mask(anomaly)

    df["CREDIT_INCOME_RATIO"] = _safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"] = _safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -days_employed / 365.25
    df["EMPLOYED_TO_AGE"] = _safe_divide(days_employed, df["DAYS_BIRTH"])

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)

    return df.drop(columns=["DAYS_EMPLOYED"])