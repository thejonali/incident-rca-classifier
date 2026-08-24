from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline

from .baseline_scoring import get_baseline_classes
from .data import normalize_text


@dataclass(frozen=True)
class FeatureContribution:
    term: str
    tfidf_value: float
    weight: float
    contribution: float


def get_class_weights(pipeline: Pipeline, class_index: int) -> np.ndarray:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.shape[0] == 1:
            return coefficients[0] if class_index == 1 else -coefficients[0]
        return coefficients[class_index]

    if hasattr(estimator, "feature_log_prob_"):
        log_probabilities = np.asarray(estimator.feature_log_prob_, dtype=float)
        return log_probabilities[class_index] - log_probabilities.mean(axis=0)

    raise ValueError("Baseline estimator does not expose feature weights for explanations")


def explain_baseline_prediction(
    pipeline: Pipeline,
    text: str,
    predicted_label: str,
    top_n: int = 8,
) -> dict:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    vectorizer = pipeline.named_steps["tfidf"]
    classes = get_baseline_classes(pipeline)
    class_index = classes.index(predicted_label)
    feature_names = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([normalize_text(text)]).tocsr()
    weights = get_class_weights(pipeline, class_index)

    nonzero_indices = vector.indices
    contributions: list[FeatureContribution] = []
    for feature_index in nonzero_indices:
        tfidf_value = float(vector[0, feature_index])
        weight = float(weights[feature_index])
        contribution = tfidf_value * weight
        if contribution > 0:
            contributions.append(
                FeatureContribution(
                    term=str(feature_names[feature_index]),
                    tfidf_value=tfidf_value,
                    weight=weight,
                    contribution=contribution,
                )
            )

    contributions.sort(key=lambda item: item.contribution, reverse=True)
    top_contributions = contributions[:top_n]

    return {
        "method": "tfidf_feature_contribution",
        "classification": predicted_label,
        "supporting_signals": [
            {
                "term": item.term,
                "tfidf_value": item.tfidf_value,
                "weight": item.weight,
                "contribution": item.contribution,
            }
            for item in top_contributions
        ],
        "important_features": [item.term for item in top_contributions],
        "note": (
            "These are model evidence signals from TF-IDF feature weights. "
            "They are not causal proof of the incident root cause."
        ),
    }
