"""Builds the scikit-learn pipeline: features -> encoding -> gradient boosting."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from scoring.config import MODEL_CATEGORICAL, MODEL_NUMERIC, RANDOM_SEED
from scoring.features import build_features


def make_pipeline(**model_params) -> Pipeline:
    params = {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": RANDOM_SEED,
    }
    params.update(model_params)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",  # a new category at serving time
        unknown_value=np.nan,                # is treated as "missing"
        encoded_missing_value=np.nan,        # instead of crashing the API
    )
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", MODEL_NUMERIC),
            ("cat", encoder, MODEL_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    n_num = len(MODEL_NUMERIC)
    categorical_mask = [False] * n_num + [True] * len(MODEL_CATEGORICAL)

    return Pipeline(
        [
            ("features", FunctionTransformer(build_features)),
            ("preprocess", preprocess),
            (
                "model",
                HistGradientBoostingClassifier(
                    categorical_features=categorical_mask, **params
                ),
            ),
        ]
    )
