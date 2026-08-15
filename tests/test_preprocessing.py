import pandas as pd

from incident_data_classification.config import INPUT_COLUMNS, LEAKY_COLUMNS
from incident_data_classification.data import build_input_text, validate_columns
from incident_data_classification.labels import LabelEncoder
from incident_data_classification.train import is_improvement


def test_input_columns_exclude_leaky_fields():
    assert not (set(INPUT_COLUMNS) & LEAKY_COLUMNS)


def test_build_input_text_normalizes_separators():
    row = pd.Series(
        {
            "title": "CPU Spike Detected",
            "affected_services": "cache-service",
            "primary_affected_service": "cache-service",
            "anomaly_types_detected": "cpu_spike|memory_spike",
            "severity": "P2",
            "cloud_provider": "gcp",
            "region": "us-central1",
            "environment": "production",
            "timeline_summary": "t+0m: CPU begins",
            "root_cause_description": "Disk I/O saturation",
            "contributing_factors": "Monitoring gap|Capacity planning",
        }
    )

    text = build_input_text(row)

    assert "cpu spike" in text
    assert "cpu spike memory spike" in text
    assert "|" not in text


def test_label_encoder_round_trip():
    encoder = LabelEncoder.fit(["B", "A", "B"])

    encoded = encoder.encode(["A", "B"])

    assert encoder.decode(encoded) == ["A", "B"]


def test_validate_columns_rejects_missing_required_columns():
    df = pd.DataFrame({"title": ["x"]})

    try:
        validate_columns(df)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("validate_columns should reject missing columns")


def test_is_improvement_handles_loss_and_score_metrics():
    assert is_improvement("val_macro_f1", current=0.51, best=0.50, min_delta=0.001)
    assert not is_improvement("val_macro_f1", current=0.5005, best=0.50, min_delta=0.001)
    assert is_improvement("val_loss", current=0.48, best=0.50, min_delta=0.001)
    assert not is_improvement("val_loss", current=0.4995, best=0.50, min_delta=0.001)
