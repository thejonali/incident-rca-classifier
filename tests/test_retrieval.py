from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from incident_data_classification.predict_with_retrieval import build_response
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


def test_predict_with_retrieval_response_includes_explanation_and_evidence(tmp_path: Path):
    models_dir = tmp_path / "models"
    artifact_dir = models_dir / "baselines" / "linear_svm" / "alert_only"
    artifact_dir.mkdir(parents=True)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("model", LinearSVC(class_weight="balanced", dual="auto")),
        ]
    )
    texts = [
        "checkout traffic spike overload high error",
        "checkout traffic spike hpa thrashing overload",
        "worker memory leak heap growth restart",
        "worker memory leak file descriptor leak",
    ]
    labels = ["TRAFFIC_OVERLOAD", "TRAFFIC_OVERLOAD", "RESOURCE_LEAK", "RESOURCE_LEAK"]
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, artifact_dir / "model.joblib")

    retrieval_dir = models_dir / "retrieval" / "alert_only"
    retrieval_index = build_retrieval_index(make_incidents(), feature_profile="alert_only")
    save_retrieval_index(retrieval_dir / "index.joblib", retrieval_index)
    args = SimpleNamespace(
        models_dir=models_dir,
        model="linear_svm",
        feature_profile="alert_only",
        top_k=1,
        top_features=3,
        confidence_threshold=None,
    )

    response = build_response(args, "checkout traffic spike high error")

    assert response["classification"] == "TRAFFIC_OVERLOAD"
    assert response["explanation"]["supporting_signals"]
    assert response["evidence"]["model_supporting_signals"]
    assert response["similar_incidents"][0]["incident_id"] == "INC-002"
    assert response["remedy_source"] == "INC-002"
