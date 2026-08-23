import pytest

from incident_data_classification.train_baseline import BASELINE_MODELS, make_baseline_pipeline


def test_make_baseline_pipeline_supports_expected_models():
    for model_name in BASELINE_MODELS:
        pipeline = make_baseline_pipeline(model_name)

        assert list(pipeline.named_steps) == ["tfidf", "model"]


def test_make_baseline_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported baseline model"):
        make_baseline_pipeline("unknown")
