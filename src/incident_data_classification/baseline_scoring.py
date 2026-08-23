from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


def load_baseline_pipeline(models_dir: Path, model_name: str, feature_profile: str) -> Pipeline:
    model_path = models_dir / "baselines" / model_name / feature_profile / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_path}. Train it first.")
    return joblib.load(model_path)


def get_baseline_classes(pipeline: Pipeline) -> list[str]:
    return [str(label) for label in pipeline.classes_.tolist()]


def get_baseline_scores(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
    elif hasattr(pipeline, "predict_log_proba"):
        scores = pipeline.predict_log_proba(texts)
    elif hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        scores = np.log(np.clip(probabilities, 1e-12, 1.0))
    else:
        raise ValueError("Baseline pipeline does not expose decision scores or probabilities")

    scores_array = np.asarray(scores, dtype=float)
    if scores_array.ndim == 1:
        scores_array = np.column_stack([-scores_array, scores_array])
    return scores_array
