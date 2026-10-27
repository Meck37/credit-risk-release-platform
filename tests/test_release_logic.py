"""Tests for release decisions that do not need a running server."""

from release.canary_analysis import decide, gather
from release.validate_model import run_checks


def _metrics(auc=0.75, nan=0, latency=10.0, slice_auc=0.72):
    return {"auc": auc, "nan_scores": nan, "p95_latency_ms": latency,
            "slices": {"AGE_BAND=<30": {"rows": 900, "auc": slice_auc}}}


def test_gate_passes_good_candidate():
    checks = run_checks(_metrics(), _metrics(auc=0.751), holdout_hash_ok=True)
    assert all(c["passed"] for c in checks)


def test_gate_blocks_candidate_worse_than_champion():
    checks = run_checks(_metrics(auc=0.74), _metrics(auc=0.76), holdout_hash_ok=True)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert failed == {"not_worse_than_champion"}


def test_gate_blocks_weak_slice_and_tampered_holdout():
    checks = run_checks(_metrics(slice_auc=0.55), None, holdout_hash_ok=False)
    failed = {c["check"] for c in checks if not c["passed"]}
    assert "holdout_file_matches_training_record" in failed
    assert "slice_auc[AGE_BAND=<30]" in failed


def _canary_data(canary_errors=0, canary_p95=0.03, canary_requests=200, canary_mean=0.10):
    return {
        "requests": {"stable": 1800, "canary": canary_requests},
        "errors": {"stable": 0, "canary": canary_errors},
        "p95": {"stable": 0.025, "canary": canary_p95},
        "mean_score": {"stable": 0.10, "canary": canary_mean},
    }


def test_healthy_canary_is_promoted():
    passed, _ = decide(_canary_data())
    assert passed


def test_erroring_canary_is_rolled_back():
    passed, checks = decide(_canary_data(canary_errors=10))
    assert not passed
    assert not next(c for c in checks if c["check"] == "canary_error_rate")["passed"]


def test_slow_canary_is_rolled_back():
    passed, _ = decide(_canary_data(canary_p95=0.2))
    assert not passed


def test_canary_without_traffic_is_never_promoted():
    passed, _ = decide(_canary_data(canary_requests=3))
    assert not passed


def test_gather_builds_queries_for_requested_env():
    seen = []

    def fake_query(promql):
        seen.append(promql)
        return {"stable": 1.0, "canary": 1.0}

    gather(fake_query, env="prod", window="5m")
    assert len(seen) == 4
    assert all('env="prod"' in q for q in seen)
