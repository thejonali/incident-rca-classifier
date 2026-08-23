from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_HARD_CASES_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    ROOT_CAUSE_CATEGORIES,
)
from .data import normalize_text
from .train_baseline import BASELINE_MODELS, save_json


REQUIRED_HARD_CASE_FIELDS = {
    "id",
    "input",
    "expected_category",
    "difficulty",
    "scenario_type",
    "notes",
}


def load_hard_cases(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Hard evaluation set not found: {path}")

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Hard evaluation set must be a JSON list")

    seen_ids: set[str] = set()
    allowed_categories = set(ROOT_CAUSE_CATEGORIES)
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Hard case at index {index} must be an object")

        missing = REQUIRED_HARD_CASE_FIELDS - set(case)
        if missing:
            raise ValueError(f"Hard case at index {index} is missing fields: {sorted(missing)}")

        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"Duplicate hard case id: {case_id}")
        seen_ids.add(case_id)

        if case["expected_category"] not in allowed_categories:
            raise ValueError(f"Hard case {case_id} has unsupported category: {case['expected_category']}")

        if not str(case["input"]).strip():
            raise ValueError(f"Hard case {case_id} has empty input")

    return cases


def build_hard_case_report(
    cases: list[dict[str, str]],
    predictions: list[str],
    model_name: str,
    feature_profile: str,
    inference_seconds: float,
) -> dict:
    y_true = [case["expected_category"] for case in cases]
    labels = list(ROOT_CAUSE_CATEGORIES)
    failures = [
        {
            "id": case["id"],
            "expected_category": expected,
            "predicted_category": predicted,
            "scenario_type": case["scenario_type"],
            "input": case["input"],
            "notes": case["notes"],
        }
        for case, expected, predicted in zip(cases, y_true, predictions, strict=True)
        if expected != predicted
    ]

    return {
        "benchmark": "hard_evaluation_set",
        "model_type": model_name,
        "feature_profile": feature_profile,
        "case_count": len(cases),
        "inference_latency_ms": (inference_seconds / max(1, len(cases))) * 1000,
        "test_accuracy": accuracy_score(y_true, predictions),
        "test_macro_f1": f1_score(y_true, predictions, average="macro"),
        "test_weighted_f1": f1_score(y_true, predictions, average="weighted"),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved baseline classifier on hard incident cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_HARD_CASES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--model", choices=BASELINE_MODELS, default="linear_svm")
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES, default=DEFAULT_FEATURE_PROFILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_hard_cases(args.cases)

    artifact_dir = args.models_dir / "baselines" / args.model / args.feature_profile
    model_path = artifact_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_path}. Train it first.")

    pipeline = joblib.load(model_path)
    texts = [normalize_text(case["input"]) for case in cases]

    start = time.perf_counter()
    predictions = pipeline.predict(texts).tolist()
    inference_seconds = time.perf_counter() - start

    report = build_hard_case_report(
        cases=cases,
        predictions=predictions,
        model_name=args.model,
        feature_profile=args.feature_profile,
        inference_seconds=inference_seconds,
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"hard_cases_{args.model}_{args.feature_profile}_metrics.json"
    save_json(report_path, report)

    print("benchmark             model       feature_profile  cases  accuracy  macro_f1  weighted_f1  failures")
    print("--------------------  ----------  ---------------  -----  --------  --------  -----------  --------")
    print(
        f"hard_evaluation_set   {args.model:<10}  {args.feature_profile:<15}  "
        f"{report['case_count']:>5}  {report['test_accuracy']:>8.3f}  "
        f"{report['test_macro_f1']:>8.3f}  {report['test_weighted_f1']:>11.3f}  "
        f"{len(report['failures']):>8}"
    )
    print(f"Saved hard-set report to {report_path}")


if __name__ == "__main__":
    main()
