import numpy as np
import pytest

from incident_data_classification.confidence import (
    abstention_summary,
    evaluate_confidence,
    fit_temperature,
    labels_to_indices,
    risk_coverage_curve,
    select_threshold_for_coverage,
    softmax,
)


def test_softmax_temperature_preserves_shape_and_normalizes_rows():
    scores = np.array([[3.0, 1.0, -1.0], [0.0, 2.0, 4.0]])

    probabilities = softmax(scores, temperature=2.0)

    assert probabilities.shape == scores.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_fit_temperature_uses_validation_labels_and_returns_positive_temperature():
    classes = ["A", "B"]
    scores = np.array([[3.0, 1.0], [2.0, 0.0], [0.0, 2.0], [1.0, 3.0]])
    labels = ["A", "A", "B", "B"]

    temperature, validation_nll = fit_temperature(scores, labels, classes, candidates=25)

    assert temperature > 0
    assert validation_nll < 1.0


def test_evaluate_confidence_reports_ece_and_brier_score():
    classes = ["A", "B"]
    labels = ["A", "B", "B"]
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])

    metrics, bins = evaluate_confidence(probabilities, labels, classes, n_bins=5)

    assert metrics.accuracy == pytest.approx(2 / 3)
    assert 0 <= metrics.ece <= 1
    assert metrics.brier_score > 0
    assert len(bins) == 5


def test_abstention_summary_tracks_accepted_and_rejected_predictions():
    classes = ["A", "B"]
    labels = ["A", "B", "B"]
    probabilities = np.array([[0.8, 0.2], [0.55, 0.45], [0.1, 0.9]])

    summary = abstention_summary(probabilities, labels, classes, threshold=0.75)

    assert summary["accepted_count"] == 2
    assert summary["rejected_count"] == 1
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["accepted_accuracy"] == pytest.approx(1.0)


def test_risk_coverage_curve_reports_requested_coverages():
    classes = ["A", "B"]
    labels = ["A", "B", "B", "A"]
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.55, 0.45], [0.6, 0.4]])

    rows = risk_coverage_curve(probabilities, labels, classes, target_coverages=(1.0, 0.5))

    assert [row["target_coverage"] for row in rows] == [1.0, 0.5]
    assert rows[0]["accepted_count"] == 4
    assert rows[1]["accepted_count"] == 2


def test_labels_to_indices_rejects_unknown_labels():
    with pytest.raises(ValueError, match="not present"):
        labels_to_indices(["A", "C"], ["A", "B"])


def test_select_threshold_rejects_invalid_coverage():
    with pytest.raises(ValueError, match="target_coverage"):
        select_threshold_for_coverage(np.array([0.9]), 0.0)
