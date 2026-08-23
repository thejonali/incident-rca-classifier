from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    TARGET_COLUMN,
)
from .data import get_feature_columns, load_incidents, split_dataset
from .dataset import resolve_incidents_csv


BASELINE_MODELS = ("logistic_regression", "linear_svm", "naive_bayes")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_baseline_pipeline(model_name: str) -> Pipeline:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_name == "linear_svm":
        estimator = LinearSVC(class_weight="balanced", dual="auto")
    elif model_name == "naive_bayes":
        estimator = MultinomialNB()
    else:
        supported = ", ".join(BASELINE_MODELS)
        raise ValueError(f"Unsupported baseline model {model_name!r}. Supported models: {supported}")

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("model", estimator),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF classical ML incident classifier")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before training.")
    parser.add_argument("--model", choices=BASELINE_MODELS, required=True)
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Input field profile used to build incident text.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    print(f"Using dataset: {csv_path}")
    print(f"Using feature profile: {args.feature_profile}")

    df = load_incidents(csv_path, max_rows=args.max_rows, feature_profile=args.feature_profile)
    splits = split_dataset(df)
    pipeline = make_baseline_pipeline(args.model)

    start = time.perf_counter()
    pipeline.fit(splits.x_train, splits.y_train)
    training_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    test_pred = pipeline.predict(splits.x_test)
    inference_seconds = time.perf_counter() - inference_start
    inference_latency_ms = (inference_seconds / max(1, len(splits.x_test))) * 1000

    labels = sorted(df[TARGET_COLUMN].unique().tolist())
    metrics = {
        "model_type": args.model,
        "model_family": "tfidf_baseline",
        "feature_profile": args.feature_profile,
        "feature_columns": get_feature_columns(args.feature_profile),
        "rows_used": int(len(df)),
        "classes": labels,
        "class_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "training_seconds": training_seconds,
        "inference_latency_ms": inference_latency_ms,
        "test_accuracy": accuracy_score(splits.y_test, test_pred),
        "test_macro_f1": f1_score(splits.y_test, test_pred, average="macro"),
        "test_weighted_f1": f1_score(splits.y_test, test_pred, average="weighted"),
        "test_macro_precision": precision_score(splits.y_test, test_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(splits.y_test, test_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            splits.y_test,
            test_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(splits.y_test, test_pred, labels=labels).tolist(),
        "hyperparameters": {
            "max_rows": args.max_rows,
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 2,
        },
    }

    artifact_dir = args.models_dir / "baselines" / args.model / args.feature_profile
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "model.joblib")
    save_json(artifact_dir / "metrics.json", metrics)
    save_json(args.reports_dir / f"baseline_{args.model}_{args.feature_profile}_metrics.json", metrics)

    print(f"\nSaved {args.model} artifacts to {artifact_dir}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Inference latency: {inference_latency_ms:.3f} ms/incident")
    print(f"Training time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()
