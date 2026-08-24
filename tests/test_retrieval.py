from pathlib import Path

import pandas as pd
import pytest

from incident_data_classification.retrieval import (
    build_retrieval_index,
    load_retrieval_index,
    retrieve_similar_incidents,
    save_retrieval_index,
)


def make_incidents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "INC-001",
                "root_cause_category": "RESOURCE_LEAK",
                "input_text": "worker memory leak heap growth restart loop",
                "remediation_that_worked": "Restart workers and patch leaked client handles.",
                "prevention_recommendation": "Add heap alerts.",
                "title": "Worker memory leak",
                "affected_services": "worker-service",
                "primary_affected_service": "worker-service",
                "anomaly_types_detected": "memory_leak",
            },
            {
                "incident_id": "INC-002",
                "root_cause_category": "TRAFFIC_OVERLOAD",
                "input_text": "checkout traffic spike hpa thrashing high error rate",
                "remediation_that_worked": "Scale checkout workers and enable rate limiting.",
                "prevention_recommendation": "Tune autoscaling policies.",
                "title": "Checkout traffic spike",
                "affected_services": "checkout-service",
                "primary_affected_service": "checkout-service",
                "anomaly_types_detected": "traffic_spike_overload",
            },
            {
                "incident_id": "INC-003",
                "root_cause_category": "DEPENDENCY_FAILURE",
                "input_text": "payment dependency timeout retries checkout errors",
                "remediation_that_worked": "Fail over to secondary payment provider.",
                "prevention_recommendation": "Add provider health routing.",
                "title": "Payment dependency timeout",
                "affected_services": "payment-service",
                "primary_affected_service": "payment-service",
                "anomaly_types_detected": "dependency_timeout",
            },
        ]
    )


def test_retrieve_similar_incidents_returns_ranked_evidence():
    index = build_retrieval_index(make_incidents(), feature_profile="alert_only")

    matches = retrieve_similar_incidents(index, "checkout traffic spike high error", top_k=2)

    assert matches[0]["incident_id"] == "INC-002"
    assert matches[0]["similarity"] > matches[1]["similarity"]
    assert matches[0]["remediation"] == "Scale checkout workers and enable rate limiting."


def test_retrieve_similar_incidents_can_filter_by_category():
    index = build_retrieval_index(make_incidents(), feature_profile="alert_only")

    matches = retrieve_similar_incidents(
        index,
        "checkout traffic spike high error",
        category="RESOURCE_LEAK",
        top_k=3,
    )

    assert [match["root_cause_category"] for match in matches] == ["RESOURCE_LEAK"]


def test_retrieve_similar_incidents_returns_empty_for_missing_category():
    index = build_retrieval_index(make_incidents(), feature_profile="alert_only")

    assert retrieve_similar_incidents(index, "anything", category="SECURITY_INCIDENT") == []


def test_retrieve_similar_incidents_rejects_invalid_top_k():
    index = build_retrieval_index(make_incidents(), feature_profile="alert_only")

    with pytest.raises(ValueError, match="top_k"):
        retrieve_similar_incidents(index, "checkout", top_k=0)


def test_retrieval_index_round_trips(tmp_path: Path):
    path = tmp_path / "index.joblib"
    index = build_retrieval_index(make_incidents(), feature_profile="alert_only")

    save_retrieval_index(path, index)
    loaded = load_retrieval_index(path)

    matches = retrieve_similar_incidents(loaded, "payment timeout retries", top_k=1)
    assert matches[0]["incident_id"] == "INC-003"
