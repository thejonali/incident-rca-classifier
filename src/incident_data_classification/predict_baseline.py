from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline_scoring import get_baseline_classes, get_baseline_scores, load_baseline_pipeline
from .confidence import softmax
from .config import DEFAULT_FEATURE_PROFILE, DEFAULT_MODELS_DIR, FEATURE_PROFILES
from .data import normalize_text
from .train_baseline import BASELINE_MODELS


SAMPLE_TEXT = (
    "checkout-service production sev2 traffic_spike_overload hpa_thrashing "
    "high_error_rate elevated_requests gateway saturation"
)


def load_calibration(artifact_dir: Path) -> dict | None:
    calibration_path = artifact_dir / "calibration.json"
    if not calibration_path.exists():
        return None
    return json.loads(calibration_path.read_text(encoding="utf-8"))


def predict_one(
    text: str,
    models_dir: Path,
    model_name: str,
    feature_profile: str,
    confidence_threshold: float | None = None,
) -> dict:
    artifact_dir = models_dir / "baselines" / model_name / feature_profile
    pipeline = load_baseline_pipeline(models_dir, model_name, feature_profile)
    classes = get_baseline_classes(pipeline)
    calibration = load_calibration(artifact_dir)

    threshold_source = "argument"
    if confidence_threshold is None and calibration is not None:
        confidence_threshold = float(calibration["threshold"])
        threshold_source = "calibration"
    elif confidence_threshold is None:
        confidence_threshold = 0.75
        threshold_source = "default"

    temperature = float(calibration["temperature"]) if calibration is not None else 1.0
    scores = get_baseline_scores(pipeline, [normalize_text(text)])
    probabilities = softmax(scores, temperature=temperature)[0]
    top_index = int(np.argmax(probabilities))
    confidence = float(probabilities[top_index])
    top_indices = np.argsort(probabilities)[::-1][:3]

    return {
        "model": model_name,
        "feature_profile": feature_profile,
        "classification": classes[top_index],
        "confidence": confidence,
        "requires_human_review": confidence < confidence_threshold,
        "confidence_threshold": confidence_threshold,
        "threshold_source": threshold_source,
        "temperature": temperature,
        "top3": [
            {
                "classification": classes[int(index)],
                "confidence": float(probabilities[int(index)]),
            }
            for index in top_indices
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated TF-IDF baseline incident predictions")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", choices=BASELINE_MODELS, default="linear_svm")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Feature profile artifact to load.",
    )
    parser.add_argument("--text", type=str, default=None, help="Incident text. Use -1 for built-in sample.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override the calibrated human-review threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text
    if text is None:
        text = input("Enter incident description, or -1 for sample incident: ").strip()
    if text == "-1":
        text = SAMPLE_TEXT

    result = predict_one(
        text=text,
        models_dir=args.models_dir,
        model_name=args.model,
        feature_profile=args.feature_profile,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
