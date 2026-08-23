import pandas as pd
import torch

from incident_data_classification.config import (
    FEATURE_PROFILE_ALERT_ONLY,
    FEATURE_PROFILE_EARLY_INCIDENT,
    FEATURE_PROFILE_POSTMORTEM,
    FEATURE_PROFILE_COLUMNS,
    INCIDENT_TIME_EXCLUDED_COLUMNS,
    INPUT_COLUMNS,
    LEAKY_COLUMNS,
)
from incident_data_classification.data import build_input_text, get_feature_columns, validate_columns
from incident_data_classification.labels import LabelEncoder
from incident_data_classification.train import is_improvement, make_class_weight_tensor


def test_input_columns_exclude_leaky_fields():
    assert not (set(INPUT_COLUMNS) & LEAKY_COLUMNS)


def make_incident_row() -> pd.Series:
    return pd.Series(
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


def test_incident_time_profiles_exclude_post_investigation_fields():
    for feature_profile in [FEATURE_PROFILE_ALERT_ONLY, FEATURE_PROFILE_EARLY_INCIDENT]:
        assert not (set(FEATURE_PROFILE_COLUMNS[feature_profile]) & INCIDENT_TIME_EXCLUDED_COLUMNS)


def test_postmortem_profile_keeps_richer_comparison_fields():
    columns = set(get_feature_columns(FEATURE_PROFILE_POSTMORTEM))

    assert "timeline_summary" in columns
    assert "root_cause_description" in columns
    assert "contributing_factors" in columns


def test_build_input_text_normalizes_separators():
    row = make_incident_row()

    text = build_input_text(row)

    assert "cpu spike" in text
    assert "cpu spike memory spike" in text
    assert "|" not in text


def test_early_incident_input_text_excludes_investigation_only_details():
    row = make_incident_row()

    text = build_input_text(row, feature_profile=FEATURE_PROFILE_EARLY_INCIDENT)

    assert "gcp" in text
    assert "us-central1" in text
    assert "disk i/o saturation" not in text
    assert "monitoring gap" not in text
    assert "t+0m" not in text


def test_postmortem_input_text_includes_investigation_details():
    row = make_incident_row()

    text = build_input_text(row, feature_profile=FEATURE_PROFILE_POSTMORTEM)

    assert "disk i/o saturation" in text
    assert "monitoring gap capacity planning" in text


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


def test_balanced_class_weights_are_larger_for_minority_classes():
    weights = make_class_weight_tensor([0, 0, 0, 1], num_classes=2, device=torch.device("cpu"))

    assert weights.shape == (2,)
    assert weights[1] > weights[0]
