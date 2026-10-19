import numpy as np
import pandas as pd

from scoring.config import (
    DAYS_EMPLOYED_SENTINEL,
    ENGINEERED_FEATURES,
    EXCLUDED_ATTRIBUTES,
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    RAW_INPUTS,
)
from scoring.features import build_features


def _row(**overrides):
    base = {c: 1.0 for c in RAW_INPUTS}
    base.update({"NAME_CONTRACT_TYPE": "Cash loans", "NAME_INCOME_TYPE": "Working",
                 "NAME_EDUCATION_TYPE": "Higher education",
                 "NAME_HOUSING_TYPE": "House / apartment", "FLAG_OWN_CAR": "N",
                 "FLAG_OWN_REALTY": "Y", "DAYS_BIRTH": -10000, "DAYS_EMPLOYED": -1000,
                 "AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 300000, "AMT_ANNUITY": 15000,
                 "AMT_GOODS_PRICE": 270000, "EXT_SOURCE_1": 0.2, "EXT_SOURCE_2": 0.4,
                 "EXT_SOURCE_3": 0.6})
    base.update(overrides)
    return pd.DataFrame([base])


def test_ratios_are_computed_correctly():
    out = build_features(_row())
    assert out.loc[0, "CREDIT_INCOME_RATIO"] == 3.0
    assert out.loc[0, "ANNUITY_INCOME_RATIO"] == 0.15
    assert out.loc[0, "CREDIT_TERM"] == 0.05
    assert np.isclose(out.loc[0, "EXT_SOURCE_MEAN"], 0.4)
    assert out.loc[0, "EXT_SOURCE_MIN"] == 0.2


def test_unemployed_placeholder_becomes_flag_not_number():
    out = build_features(_row(DAYS_EMPLOYED=DAYS_EMPLOYED_SENTINEL))
    assert out.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1.0
    assert np.isnan(out.loc[0, "EMPLOYED_YEARS"])


def test_zero_income_gives_missing_not_infinity():
    out = build_features(_row(AMT_INCOME_TOTAL=0))
    assert np.isnan(out.loc[0, "CREDIT_INCOME_RATIO"])


def test_input_frame_is_not_modified():
    raw = _row()
    before = raw.copy()
    build_features(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_all_model_columns_are_produced():
    out = build_features(_row())
    for column in MODEL_NUMERIC + MODEL_CATEGORICAL:
        assert column in out.columns
    for column in ENGINEERED_FEATURES:
        assert column in out.columns


def test_protected_attributes_are_never_model_inputs():
    used = set(RAW_INPUTS) | set(MODEL_NUMERIC) | set(MODEL_CATEGORICAL)
    assert not used & set(EXCLUDED_ATTRIBUTES)