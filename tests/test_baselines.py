import pytest

from incident_data_classification.baseline_scoring import get_baseline_scores
from incident_data_classification.train_baseline import BASELINE_MODELS, make_baseline_pipeline


def test_make_baseline_pipeline_supports_expected_models():
    for model_name in BASELINE_MODELS:
        pipeline = make_baseline_pipeline(model_name)

        assert list(pipeline.named_steps) == ["tfidf", "model"]


def test_make_baseline_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported baseline model"):
        make_baseline_pipeline("unknown")


def test_get_baseline_scores_returns_one_score_per_class():
    pipeline = make_baseline_pipeline("linear_svm")
    texts = [
        "checkout high error rate traffic spike",
        "worker memory leak heap growth",
        "checkout traffic spike high error rate",
        "worker heap memory leak",
    ]
    labels = ["TRAFFIC_OVERLOAD", "RESOURCE_LEAK", "TRAFFIC_OVERLOAD", "RESOURCE_LEAK"]
    pipeline.fit(texts, labels)

    scores = get_baseline_scores(pipeline, ["checkout traffic spike"])

    assert scores.shape == (1, 2)
