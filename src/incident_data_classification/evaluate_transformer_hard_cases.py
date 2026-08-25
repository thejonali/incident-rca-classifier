from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_HARD_CASES_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    ROOT_CAUSE_CATEGORIES,
)
from .evaluate_hard_cases import get_case_text, load_hard_cases, report_stem
from .model import get_device
from .train import save_json
from .transformer_utils import (
    TransformerIncidentDataset,
    load_transformer_artifacts,
    resolve_transformer_artifact_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved transformer on hard incident cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_HARD_CASES_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--artifact-name", type=str, default="distilbert")
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES, default=DEFAULT_FEATURE_PROFILE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def predict_cases(
    texts: list[str],
    artifact_name: str,
    models_dir: Path,
    feature_profile: str,
    batch_size: int,
    max_length: int,
    prefer_mps: bool,
) -> tuple[list[str], float]:
    device = get_device(prefer_mps=prefer_mps)
    artifact_dir = resolve_transformer_artifact_dir(models_dir, artifact_name, feature_profile)
    model, tokenizer, label_encoder = load_transformer_artifacts(artifact_dir, device)
    dataset = TransformerIncidentDataset(texts, labels=None, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[int] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
    inference_seconds = time.perf_counter() - start
    return label_encoder.decode(predictions), inference_seconds


def main() -> None:
    args = parse_args()
    cases = load_hard_cases(args.cases)
    texts = [get_case_text(case, args.feature_profile) for case in cases]
    predictions, inference_seconds = predict_cases(
        texts=texts,
        artifact_name=args.artifact_name,
        models_dir=args.models_dir,
        feature_profile=args.feature_profile,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prefer_mps=args.prefer_mps,
    )
    y_true = [case["expected_category"] for case in cases]
    labels = list(ROOT_CAUSE_CATEGORIES)
    failures = [
        {
            "id": case["id"],
            "expected_category": expected,
            "predicted_category": predicted,
            "scenario_type": case["scenario_type"],
            "input": case.get("input") or get_case_text(case, args.feature_profile),
            "notes": case["notes"],
        }
        for case, expected, predicted in zip(cases, y_true, predictions, strict=True)
        if expected != predicted
    ]

    report = {
        "benchmark": "hard_evaluation_set",
        "model_type": args.artifact_name,
        "model_family": "transformer",
        "feature_profile": args.feature_profile,
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

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"{report_stem(args.cases, args.artifact_name, args.feature_profile)}.json"
    save_json(report_path, report)

    print(f"Saved hard-case transformer report to {report_path}")
    print(f"Accuracy: {report['test_accuracy']:.3f}")
    print(f"Macro F1: {report['test_macro_f1']:.3f}")
    print(f"Weighted F1: {report['test_weighted_f1']:.3f}")
    print(f"Inference latency: {report['inference_latency_ms']:.3f} ms/incident")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
