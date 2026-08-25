import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from incident_data_classification.api.app import ApiSettings, IncidentClassificationRequest, build_incident_text, create_app
from incident_data_classification.retrieval import build_retrieval_index, save_retrieval_index


def write_api_artifacts(models_dir: Path) -> None:
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
    (artifact_dir / "calibration.json").write_text(
        json.dumps({"temperature": 1.0, "threshold": 0.10}),
        encoding="utf-8",
    )

    incidents = pd.DataFrame(
        [
            {
                "incident_id": "INC-001",
                "root_cause_category": "TRAFFIC_OVERLOAD",
                "input_text": "checkout traffic spike overload high error",
                "remediation_that_worked": "Scale checkout workers.",
                "prevention_recommendation": "Tune autoscaling policies.",
                "title": "Checkout traffic spike",
                "affected_services": "checkout-service",
                "primary_affected_service": "checkout-service",
                "anomaly_types_detected": "traffic_spike_overload",
            },
            {
                "incident_id": "INC-002",
                "root_cause_category": "RESOURCE_LEAK",
                "input_text": "worker memory leak heap growth restart",
                "remediation_that_worked": "Restart workers and patch leaked handles.",
                "prevention_recommendation": "Add heap alerts.",
                "title": "Worker memory leak",
                "affected_services": "worker-service",
                "primary_affected_service": "worker-service",
                "anomaly_types_detected": "memory_leak",
            },
        ]
    )
    retrieval_dir = models_dir / "retrieval" / "alert_only"
    save_retrieval_index(retrieval_dir / "index.joblib", build_retrieval_index(incidents, "alert_only"))


def make_client(models_dir: Path) -> TestClient:
    app = create_app(ApiSettings(models_dir=models_dir, feature_profile="alert_only", top_k=1, top_features=3))
    return TestClient(app)


def test_health_reports_loaded_model_and_retrieval(tmp_path: Path):
    write_api_artifacts(tmp_path)

    with make_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert response.json()["retrieval_loaded"] is True


def test_model_metadata_exposes_only_linear_svm(tmp_path: Path):
    write_api_artifacts(tmp_path)

    with make_client(tmp_path) as client:
        response = client.get("/v1/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "linear_svm"
    assert payload["feature_profile"] == "alert_only"
    assert payload["supports_abstention"] is True
    assert "TRAFFIC_OVERLOAD" in payload["classes"]


def test_classify_endpoint_returns_confidence_explanation_and_evidence(tmp_path: Path):
    write_api_artifacts(tmp_path)

    with make_client(tmp_path) as client:
        response = client.post(
            "/v1/incidents/classify",
            headers={"X-Request-ID": "request-123"},
            json={
                "title": "Checkout traffic spike",
                "severity": "SEV2",
                "affected_services": ["checkout-service"],
                "primary_affected_service": "checkout-service",
                "anomalies": ["traffic_spike_overload", "high_error_rate"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "request-123"
    assert payload["model"] == "linear_svm"
    assert payload["classification"] == "TRAFFIC_OVERLOAD"
    assert payload["inference_latency_ms"] >= 0
    assert payload["alternatives"]
    assert payload["explanation"]["supporting_signals"]
    assert payload["evidence"]["retrieved_incident_ids"] == ["INC-001"]
    assert payload["recommended_remedy"] == "Scale checkout workers."
    assert payload["remedy_source"] == "INC-001"


def test_classify_endpoint_rejects_empty_payload(tmp_path: Path):
    write_api_artifacts(tmp_path)

    with make_client(tmp_path) as client:
        response = client.post("/v1/incidents/classify", json={})

    assert response.status_code == 422


def test_classify_endpoint_returns_503_when_model_is_missing(tmp_path: Path):
    with make_client(tmp_path) as client:
        health = client.get("/health")
        response = client.post("/v1/incidents/classify", json={"text": "checkout traffic spike"})

    assert health.json()["status"] == "error"
    assert response.status_code == 503


def test_build_incident_text_accepts_raw_text_or_fields():
    raw = build_incident_text(IncidentClassificationRequest(text="Traffic_Spike | Gateway"))
    fields = build_incident_text(
        IncidentClassificationRequest(
            title="Checkout failures",
            affected_services=["checkout-service"],
            anomalies=["traffic_spike_overload"],
        )
    )

    assert raw == "traffic spike gateway"
    assert "checkout failures" in fields
    assert "traffic spike overload" in fields
