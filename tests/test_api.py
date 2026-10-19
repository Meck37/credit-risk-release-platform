def test_live_and_ready(client):
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "test"


def test_score_returns_valid_probability_and_band(client, valid_application):
    response = client.post("/v1/score", json=valid_application)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["risk_band"] in {"A", "B", "C", "D", "E"}
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["model_version"] == "test"


def test_request_id_is_echoed_for_traceability(client, valid_application):
    response = client.post("/v1/score", json=valid_application,
                           headers={"x-request-id": "trace-123"})
    assert response.json()["request_id"] == "trace-123"


def test_missing_optional_fields_still_score(client, valid_application):
    for field in ("EXT_SOURCE_1", "EXT_SOURCE_3", "AMT_ANNUITY", "OWN_CAR_AGE"):
        valid_application[field] = None
    assert client.post("/v1/score", json=valid_application).status_code == 200


def test_unknown_category_does_not_crash(client, valid_application):
    valid_application["NAME_INCOME_TYPE"] = "A brand new income type"
    assert client.post("/v1/score", json=valid_application).status_code == 200


def test_impossible_values_are_rejected(client, valid_application):
    bad_cases = [
        {"AMT_INCOME_TOTAL": -5},
        {"EXT_SOURCE_2": 1.7},
        {"DAYS_BIRTH": -50},          # a 0-year-old applicant
        {"FLAG_OWN_CAR": "maybe"},
    ]
    for change in bad_cases:
        payload = {**valid_application, **change}
        assert client.post("/v1/score", json=payload).status_code == 422, change


def test_unexpected_fields_are_rejected(client, valid_application):
    payload = {**valid_application, "CODE_GENDER": "F"}
    assert client.post("/v1/score", json=payload).status_code == 422


def test_metrics_expose_model_and_http_series(client, valid_application):
    client.post("/v1/score", json=valid_application)
    text = client.get("/metrics").text
    assert "scoring_prediction_score_bucket" in text
    assert 'scoring_model_info{' in text
    assert "http_request_duration_seconds" in text
