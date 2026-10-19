"""Shared test fixtures.

`trained_model_dir` trains a small model on synthetic data and saves it in the
same layout the Docker image uses (/app/model), so the API tests exercise the
real loading code without needing an MLflow server.
"""

import json
import os

import mlflow
import pytest

from scoring.config import MODEL_NAME, RAW_INPUTS, SKOPS_TRUSTED_TYPES, TARGET
from scoring.model_factory import make_pipeline
from training.make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture_data():
    return make_fixture(rows=4000, seed=7)


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, fixture_data):
    model_dir = tmp_path_factory.mktemp("model") / "model"
    model = make_pipeline(max_iter=60)
    model.fit(fixture_data[RAW_INPUTS], fixture_data[TARGET])
    mlflow.sklearn.save_model(model, str(model_dir), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    meta = {
        "model_name": MODEL_NAME, "model_version": "test", "run_id": "test-run",
        "git_sha": "test", "holdout_sha256": "test", "holdout_auc": "0",
        "baseline_mean_score": float(model.predict_proba(fixture_data[RAW_INPUTS])[:, 1].mean()),
        "exported_at": "test",
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta))
    return model_dir


@pytest.fixture(scope="session")
def client(trained_model_dir):
    os.environ["MODEL_DIR"] = str(trained_model_dir)
    from fastapi.testclient import TestClient

    from scoring import app as app_module

    app_module.MODEL_DIR = trained_model_dir
    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def valid_application():
    return {
        "application_id": "APP-0001",
        "AMT_INCOME_TOTAL": 202500.0,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000.0,
        "DAYS_BIRTH": -9461,
        "DAYS_EMPLOYED": -637,
        "DAYS_ID_PUBLISH": -2120,
        "DAYS_REGISTRATION": -3648.0,
        "EXT_SOURCE_1": 0.083,
        "EXT_SOURCE_2": 0.263,
        "EXT_SOURCE_3": 0.139,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 1.0,
        "OWN_CAR_AGE": None,
        "REGION_RATING_CLIENT": 2,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_HOUSING_TYPE": "House / apartment",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
    }
