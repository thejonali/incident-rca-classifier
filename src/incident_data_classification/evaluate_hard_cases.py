from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_HARD_CASES_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    ROOT_CAUSE_CATEGORIES,
)
from .data import build_input_text, get_feature_columns, normalize_text
from .predict import load_model
from .train_baseline import BASELINE_MODELS, save_json


NEURAL_MODELS = ("gru", "lstm")
SUPPORTED_MODELS = BASELINE_MODELS + NEURAL_MODELS

REQUIRED_HARD_CASE_FIELDS = {
    "id",
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

        if "input" in case:
            if not str(case["input"]).strip():
                raise ValueError(f"Hard case {case_id} has empty input")
            continue

        required_structured_columns = {
            column
            for feature_profile in FEATURE_PROFILES
            for column in get_feature_columns(feature_profile)
            if column != "early_timeline_summary"
        }
        missing_profile_columns = [column for column in sorted(required_structured_columns) if column not in case]
        if missing_profile_columns:
            raise ValueError(f"Hard case {case_id} is missing structured fields: {missing_profile_columns}")

    return cases


def get_case_text(case: dict[str, str], feature_profile: str) -> str:
    if "input" in case:
        return normalize_text(case["input"])
    return build_input_text(pd.Series(case), feature_profile=feature_profile)


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
            "input": case.get("input") or get_case_text(case, feature_profile),
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
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers on hard incident cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_HARD_CASES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--model", choices=("all",) + SUPPORTED_MODELS, default="linear_svm")
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES, default=DEFAULT_FEATURE_PROFILE)
    return parser.parse_args()


def predict_with_model(
    cases: list[dict[str, str]],
    models_dir: Path,
    model_name: str,
    feature_profile: str,
) -> list[str]:
    texts = [get_case_text(case, feature_profile) for case in cases]

    if model_name in BASELINE_MODELS:
        model_path = models_dir / "baselines" / model_name / feature_profile / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Baseline model not found at {model_path}. Train it first.")
        pipeline = joblib.load(model_path)
        return pipeline.predict(texts).tolist()

    artifact_dir = models_dir / model_name / feature_profile
    if not (artifact_dir / "model.pt").exists():
        raise FileNotFoundError(f"Neural model not found at {artifact_dir}. Train it first.")

    model, tokenizer, label_encoder, device = load_model(artifact_dir, prefer_mps=False)
    sequences = torch.tensor([tokenizer.encode(text) for text in texts], dtype=torch.long, device=device)
    with torch.no_grad():
        predictions = model(sequences).argmax(dim=1).cpu().tolist()
    return label_encoder.decode(predictions)


def report_stem(cases_path: Path, model_name: str, feature_profile: str) -> str:
    return f"{cases_path.stem}_{model_name}_{feature_profile}_metrics"


def main() -> None:
    args = parse_args()
    cases = load_hard_cases(args.cases)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    model_names = SUPPORTED_MODELS if args.model == "all" else (args.model,)
    print("benchmark             model                feature_profile  cases  accuracy  macro_f1  weighted_f1  failures")
    print("--------------------  -------------------  ---------------  -----  --------  --------  -----------  --------")
    for model_name in model_names:
        start = time.perf_counter()
        predictions = predict_with_model(cases, args.models_dir, model_name, args.feature_profile)
        inference_seconds = time.perf_counter() - start

        report = build_hard_case_report(
            cases=cases,
            predictions=predictions,
            model_name=model_name,
            feature_profile=args.feature_profile,
            inference_seconds=inference_seconds,
        )
        report_path = args.reports_dir / f"{report_stem(args.cases, model_name, args.feature_profile)}.json"
        save_json(report_path, report)

        print(
            f"hard_evaluation_set   {model_name:<19}  {args.feature_profile:<15}  "
            f"{report['case_count']:>5}  {report['test_accuracy']:>8.3f}  "
            f"{report['test_macro_f1']:>8.3f}  {report['test_weighted_f1']:>11.3f}  "
            f"{len(report['failures']):>8}"
        )


if __name__ == "__main__":
    main()
