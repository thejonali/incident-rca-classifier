import pytest

from incident_data_classification.explainability import explain_baseline_prediction
from incident_data_classification.train_baseline import make_baseline_pipeline


def fit_linear_svm_pipeline():
    pipeline = make_baseline_pipeline("linear_svm")
    texts = [
        "checkout traffic spike overload high error",
        "checkout traffic spike hpa thrashing overload",
        "worker memory leak heap growth restart",
        "worker memory leak file descriptor leak",
    ]
    labels = ["TRAFFIC_OVERLOAD", "TRAFFIC_OVERLOAD", "RESOURCE_LEAK", "RESOURCE_LEAK"]
    pipeline.fit(texts, labels)
    return pipeline


def test_explain_baseline_prediction_returns_supporting_signals():
    pipeline = fit_linear_svm_pipeline()

    explanation = explain_baseline_prediction(
        pipeline,
        "checkout traffic spike high error",
        predicted_label="TRAFFIC_OVERLOAD",
        top_n=3,
    )

    assert explanation["method"] == "tfidf_feature_contribution"
    assert explanation["classification"] == "TRAFFIC_OVERLOAD"
    assert explanation["important_features"]
    assert "not causal proof" in explanation["note"]
    assert all(item["contribution"] > 0 for item in explanation["supporting_signals"])


def test_explain_baseline_prediction_limits_top_features():
    pipeline = fit_linear_svm_pipeline()

    explanation = explain_baseline_prediction(
        pipeline,
        "checkout traffic spike high error",
        predicted_label="TRAFFIC_OVERLOAD",
        top_n=2,
    )

    assert len(explanation["supporting_signals"]) <= 2
    assert len(explanation["important_features"]) <= 2


def test_explain_baseline_prediction_rejects_invalid_top_n():
    pipeline = fit_linear_svm_pipeline()

    with pytest.raises(ValueError, match="top_n"):
        explain_baseline_prediction(pipeline, "checkout traffic", "TRAFFIC_OVERLOAD", top_n=0)


def test_explain_baseline_prediction_supports_naive_bayes_weights():
    pipeline = make_baseline_pipeline("naive_bayes")
    texts = [
        "checkout traffic spike overload high error",
        "checkout traffic spike hpa thrashing overload",
        "worker memory leak heap growth restart",
        "worker memory leak file descriptor leak",
    ]
    labels = ["TRAFFIC_OVERLOAD", "TRAFFIC_OVERLOAD", "RESOURCE_LEAK", "RESOURCE_LEAK"]
    pipeline.fit(texts, labels)

    explanation = explain_baseline_prediction(
        pipeline,
        "worker memory leak heap growth",
        predicted_label="RESOURCE_LEAK",
        top_n=3,
    )

    assert explanation["important_features"]
