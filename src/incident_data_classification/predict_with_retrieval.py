from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline_scoring import load_baseline_pipeline
from .config import DEFAULT_FEATURE_PROFILE, DEFAULT_MODELS_DIR, FEATURE_PROFILES
from .explainability import explain_baseline_prediction
from .predict_baseline import SAMPLE_TEXT, predict_one
from .retrieval import load_retrieval_index, retrieve_similar_incidents
from .train_baseline import BASELINE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify an incident and retrieve similar historical incidents")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", choices=BASELINE_MODELS, default="linear_svm")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Feature profile artifact to load.",
    )
    parser.add_argument("--text", type=str, default=None, help="Incident text. Use -1 for built-in sample.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    return parser.parse_args()


def build_response(args: argparse.Namespace, text: str) -> dict:
    classification = predict_one(
        text=text,
        models_dir=args.models_dir,
        model_name=args.model,
        feature_profile=args.feature_profile,
        confidence_threshold=args.confidence_threshold,
    )
    retrieval_path = args.models_dir / "retrieval" / args.feature_profile / "index.joblib"
    retrieval_index = load_retrieval_index(retrieval_path)
    similar_incidents = retrieve_similar_incidents(
        retrieval_index,
        text,
        category=classification["classification"],
        top_k=args.top_k,
    )
    pipeline = load_baseline_pipeline(args.models_dir, args.model, args.feature_profile)
    explanation = explain_baseline_prediction(
        pipeline,
        text,
        predicted_label=classification["classification"],
        top_n=args.top_features,
    )
    top_match = similar_incidents[0] if similar_incidents else None

    return {
        **classification,
        "explanation": explanation,
        "evidence": {
            "model_supporting_signals": explanation["important_features"],
            "retrieved_incident_ids": [match["incident_id"] for match in similar_incidents],
            "note": (
                "Evidence combines model-supporting TF-IDF signals and similar historical incidents. "
                "It is not causal proof."
            ),
        },
        "similar_incidents": similar_incidents,
        "recommended_remedy": top_match["remediation"] if top_match else None,
        "remedy_source": top_match["incident_id"] if top_match else None,
        "remedy_source_similarity": top_match["similarity"] if top_match else None,
    }


def main() -> None:
    args = parse_args()
    text = args.text
    if text is None:
        text = input("Enter incident description, or -1 for sample incident: ").strip()
    if text == "-1":
        text = SAMPLE_TEXT

    print(json.dumps(build_response(args, text), indent=2))


if __name__ == "__main__":
    main()
