from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class ConfidenceMetrics:
    accuracy: float
    average_confidence: float
    ece: float
    brier_score: float
    negative_log_likelihood: float


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scaled_scores = scores / temperature
    shifted_scores = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def labels_to_indices(labels: list[str], classes: list[str]) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(classes)}
    missing = sorted({label for label in labels if label not in class_to_index})
    if missing:
        raise ValueError(f"Labels are not present in classes: {missing}")
    return np.array([class_to_index[label] for label in labels], dtype=np.int64)


def negative_log_likelihood(probabilities: np.ndarray, true_indices: np.ndarray) -> float:
    selected = probabilities[np.arange(len(true_indices)), true_indices]
    return float(-np.mean(np.log(np.clip(selected, EPSILON, 1.0))))


def brier_score_multiclass(probabilities: np.ndarray, true_indices: np.ndarray) -> float:
    encoded = np.zeros_like(probabilities)
    encoded[np.arange(len(true_indices)), true_indices] = 1.0
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == true_indices
    bins: list[dict[str, float | int]] = []
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())
        if count:
            bin_accuracy = float(correct[mask].mean())
            bin_confidence = float(confidences[mask].mean())
        else:
            bin_accuracy = 0.0
            bin_confidence = 0.0

        weight = count / max(1, len(confidences))
        ece += weight * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
            }
        )

    return float(ece), bins


def evaluate_confidence(
    probabilities: np.ndarray,
    true_labels: list[str],
    classes: list[str],
    n_bins: int = 10,
) -> tuple[ConfidenceMetrics, list[dict[str, float | int]]]:
    true_indices = labels_to_indices(true_labels, classes)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    ece, bins = expected_calibration_error(probabilities, true_indices, n_bins=n_bins)
    metrics = ConfidenceMetrics(
        accuracy=float(np.mean(predictions == true_indices)),
        average_confidence=float(np.mean(confidences)),
        ece=ece,
        brier_score=brier_score_multiclass(probabilities, true_indices),
        negative_log_likelihood=negative_log_likelihood(probabilities, true_indices),
    )
    return metrics, bins


def fit_temperature(
    validation_scores: np.ndarray,
    validation_labels: list[str],
    classes: list[str],
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    candidates: int = 400,
) -> tuple[float, float]:
    if candidates < 2:
        raise ValueError("candidates must be at least 2")

    true_indices = labels_to_indices(validation_labels, classes)
    search_space = np.geomspace(min_temperature, max_temperature, candidates)
    best_temperature = 1.0
    best_nll = math.inf

    for temperature in search_space:
        probabilities = softmax(validation_scores, temperature=float(temperature))
        nll = negative_log_likelihood(probabilities, true_indices)
        if nll < best_nll:
            best_temperature = float(temperature)
            best_nll = nll

    return best_temperature, best_nll


def select_threshold_for_coverage(confidences: np.ndarray, target_coverage: float) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in the interval (0, 1]")

    if len(confidences) == 0:
        return 1.0

    sorted_confidences = np.sort(confidences)[::-1]
    index = min(len(sorted_confidences) - 1, max(0, math.ceil(len(sorted_confidences) * target_coverage) - 1))
    return float(sorted_confidences[index])


def abstention_summary(
    probabilities: np.ndarray,
    true_labels: list[str],
    classes: list[str],
    threshold: float,
) -> dict[str, float | int]:
    true_indices = labels_to_indices(true_labels, classes)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum())
    rejected_count = int(len(confidences) - accepted_count)
    accepted_accuracy = float(np.mean(predictions[accepted] == true_indices[accepted])) if accepted_count else 0.0

    return {
        "threshold": float(threshold),
        "coverage": accepted_count / max(1, len(confidences)),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "accepted_accuracy": accepted_accuracy,
        "abstention_rate": rejected_count / max(1, len(confidences)),
    }


def risk_coverage_curve(
    probabilities: np.ndarray,
    true_labels: list[str],
    classes: list[str],
    target_coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7),
) -> list[dict[str, float | int]]:
    confidences = np.max(probabilities, axis=1)
    rows = []
    for target_coverage in target_coverages:
        threshold = select_threshold_for_coverage(confidences, target_coverage)
        row = abstention_summary(probabilities, true_labels, classes, threshold)
        row["target_coverage"] = target_coverage
        rows.append(row)
    return rows


def write_reliability_diagram(path: Path, bins: list[dict[str, float | int]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 720
    height = 480
    margin_left = 72
    margin_right = 28
    margin_top = 54
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_width = plot_width / max(1, len(bins))

    def x_for(value: float) -> float:
        return margin_left + value * plot_width

    def y_for(value: float) -> float:
        return margin_top + (1.0 - value) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial, sans-serif" font-size="18" fill="#111827">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{x_for(0)}" y1="{y_for(0)}" x2="{x_for(1)}" y2="{y_for(1)}" stroke="#9ca3af" stroke-dasharray="5 5"/>',
    ]

    for tick in range(0, 11, 2):
        value = tick / 10
        x = x_for(value)
        y = y_for(value)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top + plot_height}" x2="{x:.1f}" y2="{margin_top + plot_height + 5}" stroke="#111827"/>')
        parts.append(f'<text x="{x - 10:.1f}" y="{margin_top + plot_height + 24}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{value:.1f}</text>')
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#111827"/>')
        parts.append(f'<text x="{margin_left - 42}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{value:.1f}</text>')

    for index, bin_row in enumerate(bins):
        lower = float(bin_row["lower"])
        confidence = float(bin_row["confidence"])
        accuracy = float(bin_row["accuracy"])
        count = int(bin_row["count"])
        x = margin_left + index * bar_width + 4
        bar_height = accuracy * plot_height
        y = margin_top + plot_height - bar_height
        fill = "#2563eb" if count else "#e5e7eb"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_width - 8):.1f}" height="{bar_height:.1f}" fill="{fill}" opacity="0.78">'
            f"<title>{lower:.1f}-{float(bin_row['upper']):.1f}: accuracy {accuracy:.3f}, confidence {confidence:.3f}, count {count}</title>"
            "</rect>"
        )
        if count:
            cx = x_for(confidence)
            cy = y_for(accuracy)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#dc2626"/>')

    parts.append(f'<text x="{margin_left + plot_width / 2 - 54:.1f}" y="{height - 16}" font-family="Arial, sans-serif" font-size="13" fill="#111827">Confidence</text>')
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2 + 42:.1f}" transform="rotate(-90 18 {margin_top + plot_height / 2 + 42:.1f})" '
        'font-family="Arial, sans-serif" font-size="13" fill="#111827">Accuracy</text>'
    )
    parts.append("</svg>")

    path.write_text("\n".join(parts), encoding="utf-8")
