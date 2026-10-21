"""Model evaluation used by training and by the release gate."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from scoring.config import GATE_MIN_SLICE_SIZE, RAW_INPUTS, TARGET

SLICE_COLUMNS = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "AGE_BAND"]


def ks_statistic(y_true, scores) -> float:
    """Kolmogorov-Smirnov: max gap between good and bad score distributions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.25
    return pd.cut(years, bins=[0, 30, 45, 60, 120], labels=["<30", "30-44", "45-59", "60+"])


def evaluate(model, df: pd.DataFrame) -> dict:
    """Return headline metrics, per-slice AUC and latency for one model."""
    X, y = df[RAW_INPUTS], df[TARGET]
    scores = model.predict_proba(X)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y, scores)),
        "ks": ks_statistic(y, scores),
        "brier": float(brier_score_loss(y, scores)),
        "mean_score": float(np.mean(scores)),
        "nan_scores": int(np.isnan(scores).sum()),
    }

    slices = {}
    frame = df.assign(AGE_BAND=age_band(df["DAYS_BIRTH"]), _score=scores)
    for column in SLICE_COLUMNS:
        for value, group in frame.groupby(column, observed=True):
            if len(group) < GATE_MIN_SLICE_SIZE or group[TARGET].nunique() < 2:
                continue  # too small to judge fairly
            slices[f"{column}={value}"] = {
                "rows": int(len(group)),
                "auc": float(roc_auc_score(group[TARGET], group["_score"])),
            }
    metrics["slices"] = slices

    # Single-row latency, the way the API calls the model.
    sample = X.head(200)
    timings = []
    for i in range(len(sample)):
        start = time.perf_counter()
        model.predict_proba(sample.iloc[[i]])
        timings.append((time.perf_counter() - start) * 1000)
    metrics["p95_latency_ms"] = float(np.percentile(timings, 95))
    return metrics