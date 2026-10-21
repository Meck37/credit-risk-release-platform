"""Central configuration shared by training, release tooling and the API.

Keeping every feature list and threshold in one place means training and
serving can never silently disagree about what the model expects.
"""

MODEL_NAME = "hc-credit-risk"
TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Raw columns the API accepts and training reads from application_train.csv.
NUMERIC_INPUTS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
    "REGION_RATING_CLIENT",
]
CATEGORICAL_INPUTS = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]
RAW_INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS

# Protected or sensitive attributes are deliberately never used as inputs.
EXCLUDED_ATTRIBUTES = ["CODE_GENDER", "NAME_FAMILY_STATUS"]

# Columns created by scoring.features.build_features.
ENGINEERED_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "GOODS_CREDIT_RATIO",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "EMPLOYED_TO_AGE",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "DAYS_EMPLOYED_ANOMALY",
]
MODEL_NUMERIC = [c for c in NUMERIC_INPUTS if c != "DAYS_EMPLOYED"] + ENGINEERED_FEATURES
MODEL_CATEGORICAL = CATEGORICAL_INPUTS

# Home Credit stores "not employed" as this placeholder value.
DAYS_EMPLOYED_SENTINEL = 365243

RANDOM_SEED = 47

# Decision policy applied to the probability of default (PD).
RISK_BANDS = [  # (upper PD bound, band, decision)
    (0.05, "A", "approve"),
    (0.10, "B", "approve"),
    (0.20, "C", "review"),
    (0.35, "D", "review"),
    (1.01, "E", "decline"),
]

# Release gate thresholds (see release/validate_model.py).
GATE_MIN_AUC = 0.70
GATE_MAX_AUC_DROP_VS_CHAMPION = 0.005
GATE_MIN_SLICE_AUC = 0.65
GATE_MIN_SLICE_SIZE = 500
GATE_MAX_P95_LATENCY_MS = 50.0

# Types the model file is allowed to contain. MLflow 3 saves scikit-learn
# models with skops, which refuses to load anything not on a reviewed list,
# so a tampered model file cannot run arbitrary code when the API loads it.
SKOPS_TRUSTED_TYPES = [
    "functools.partial",
    "scoring.features.build_features",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]