from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline_scoring import get_baseline_classes, get_baseline_scores, load_baseline_pipeline
from .confidence import (
    abstention_summary,
    evaluate_confidence,
    fit_temperature,
    risk_coverage_curve,
    select_threshold_for_coverage,
    softmax,
    write_reliability_diagram,
)
from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
)
from .data import get_feature_columns, load_incidents, split_dataset
from .dataset import resolve_incidents_csv
from .train_baseline import BASELINE_MODELS, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline confidence calibration and abstention")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before evaluation.")
    parser.add_argument("--model", choices=BASELINE_MODELS, default="linear_svm")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Input field profile used to build incident text.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help="Validation-set coverage target used to select the human-review threshold.",
    )
    return parser.parse_args()


def serialise_metrics(metrics) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy,
        "average_confidence": metrics.average_confidence,
        "expected_calibration_error": metrics.ece,
        "brier_score": metrics.brier_score,
        "negative_log_likelihood": metrics.negative_log_likelihood,
    }


def build_report(args: argparse.Namespace) -> dict:
    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    df = load_incidents(csv_path, max_rows=args.max_rows, feature_profile=args.feature_profile)
    splits = split_dataset(df)

    pipeline = load_baseline_pipeline(args.models_dir, args.model, args.feature_profile)
    classes = get_baseline_classes(pipeline)
    validation_scores = get_baseline_scores(pipeline, splits.x_val)
    test_scores = get_baseline_scores(pipeline, splits.x_test)

    raw_validation_probabilities = softmax(validation_scores)
    raw_test_probabilities = softmax(test_scores)
    temperature, validation_nll = fit_temperature(validation_scores, splits.y_val, classes)
    calibrated_validation_probabilities = softmax(validation_scores, temperature=temperature)
    calibrated_test_probabilities = softmax(test_scores, temperature=temperature)

    raw_metrics, raw_bins = evaluate_confidence(raw_test_probabilities, splits.y_test, classes, n_bins=args.bins)
    calibrated_metrics, calibrated_bins = evaluate_confidence(
        calibrated_test_probabilities,
        splits.y_test,
        classes,
        n_bins=args.bins,
    )

    validation_threshold = select_threshold_for_coverage(
        np.max(calibrated_validation_probabilities, axis=1),
        args.target_coverage,
    )

    artifact_dir = args.models_dir / "baselines" / args.model / args.feature_profile
    report_stem = f"confidence_{args.model}_{args.feature_profile}"
    raw_diagram_path = args.reports_dir / f"{report_stem}_raw_reliability.svg"
    calibrated_diagram_path = args.reports_dir / f"{report_stem}_calibrated_reliability.svg"
    write_reliability_diagram(raw_diagram_path, raw_bins, f"{args.model} {args.feature_profile} raw reliability")
    write_reliability_diagram(
        calibrated_diagram_path,
        calibrated_bins,
        f"{args.model} {args.feature_profile} calibrated reliability",
    )

    calibration_payload = {
        "model_type": args.model,
        "model_family": "tfidf_baseline",
        "feature_profile": args.feature_profile,
        "temperature": temperature,
        "threshold": validation_threshold,
        "threshold_selection": {
            "method": "validation_target_coverage",
            "target_coverage": args.target_coverage,
        },
        "classes": classes,
    }
    save_json(artifact_dir / "calibration.json", calibration_payload)

    return {
        "model_type": args.model,
        "model_family": "tfidf_baseline",
        "feature_profile": args.feature_profile,
        "feature_columns": get_feature_columns(args.feature_profile),
        "rows_used": int(len(df)),
        "validation_rows": len(splits.y_val),
        "test_rows": len(splits.y_test),
        "classes": classes,
        "temperature": temperature,
        "validation_temperature_nll": validation_nll,
        "review_threshold": validation_threshold,
        "threshold_selection": calibration_payload["threshold_selection"],
        "raw": serialise_metrics(raw_metrics),
        "calibrated": serialise_metrics(calibrated_metrics),
        "abstention_at_review_threshold": abstention_summary(
            calibrated_test_probabilities,
            splits.y_test,
            classes,
            validation_threshold,
        ),
        "risk_coverage_curve": risk_coverage_curve(calibrated_test_probabilities, splits.y_test, classes),
        "reliability_diagrams": {
            "raw": str(raw_diagram_path),
            "calibrated": str(calibrated_diagram_path),
        },
        "reliability_bins": {
            "raw": raw_bins,
            "calibrated": calibrated_bins,
        },
    }


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(args)
    report_path = args.reports_dir / f"confidence_{args.model}_{args.feature_profile}_metrics.json"
    save_json(report_path, report)

    print(f"Saved confidence report to {report_path}")
    print(f"Temperature: {report['temperature']:.3f}")
    print(f"Review threshold: {report['review_threshold']:.3f}")
    print(
        "Raw confidence: "
        f"ECE={report['raw']['expected_calibration_error']:.3f} "
        f"Brier={report['raw']['brier_score']:.3f} "
        f"NLL={report['raw']['negative_log_likelihood']:.3f}"
    )
    print(
        "Calibrated confidence: "
        f"ECE={report['calibrated']['expected_calibration_error']:.3f} "
        f"Brier={report['calibrated']['brier_score']:.3f} "
        f"NLL={report['calibrated']['negative_log_likelihood']:.3f}"
    )
    abstention = report["abstention_at_review_threshold"]
    print(
        "At threshold: "
        f"coverage={abstention['coverage']:.3f} "
        f"accepted_accuracy={abstention['accepted_accuracy']:.3f} "
        f"rejected={abstention['rejected_count']}"
    )
    print(json.dumps(report["risk_coverage_curve"], indent=2))


if __name__ == "__main__":
    main()
