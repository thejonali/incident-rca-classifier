from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODELS_DIR, DEFAULT_REPORTS_DIR, FEATURE_PROFILES
from .train_baseline import BASELINE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved GRU and LSTM metrics across feature profiles")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser.parse_args()


def load_metrics(models_dir: Path, model_name: str) -> dict | None:
    metrics_path = models_dir / model_name / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_profile_metrics(models_dir: Path, reports_dir: Path, model_name: str, feature_profile: str) -> dict | None:
    metrics_path = models_dir / model_name / feature_profile / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    report_path = reports_dir / f"{model_name}_{feature_profile}_metrics.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    return None


def load_baseline_metrics(models_dir: Path, reports_dir: Path, model_name: str, feature_profile: str) -> dict | None:
    metrics_path = models_dir / "baselines" / model_name / feature_profile / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    report_path = reports_dir / f"baseline_{model_name}_{feature_profile}_metrics.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    return None


def format_metrics_row(
    model_name: str,
    feature_profile: str,
    metrics: dict | None,
) -> tuple[str, str, str, str, str, str, str]:
    if metrics is None:
        return (model_name, feature_profile, "missing", "-", "-", "-", "-")

    train_time = f"{metrics['training_seconds']:.1f}s"
    if "inference_latency_ms" in metrics:
        train_time = f"{train_time}/{metrics['inference_latency_ms']:.3f}ms"

    return (
        model_name,
        metrics.get("feature_profile", feature_profile),
        str(metrics["rows_used"]),
        f"{metrics['test_accuracy']:.3f}",
        f"{metrics['test_macro_f1']:.3f}",
        f"{metrics['test_weighted_f1']:.3f}",
        train_time,
    )


def main() -> None:
    args = parse_args()
    rows = []
    for model_name in ["gru", "lstm"]:
        for feature_profile in FEATURE_PROFILES:
            metrics = load_profile_metrics(args.models_dir, args.reports_dir, model_name, feature_profile)
            rows.append(format_metrics_row(model_name.upper(), feature_profile, metrics))

        legacy_metrics = load_metrics(args.models_dir, model_name)
        if legacy_metrics is not None:
            rows.append(format_metrics_row(model_name.upper(), "legacy", legacy_metrics))

    for model_name in BASELINE_MODELS:
        display_name = model_name.upper()
        for feature_profile in FEATURE_PROFILES:
            metrics = load_baseline_metrics(args.models_dir, args.reports_dir, model_name, feature_profile)
            rows.append(format_metrics_row(display_name, feature_profile, metrics))

    print("model                feature_profile  rows  accuracy  macro_f1  weighted_f1  train_time/infer")
    print("-------------------  ---------------  ----  --------  --------  -----------  ----------------")
    for row in rows:
        print(f"{row[0]:<19}  {row[1]:<15}  {row[2]:>4}  {row[3]:>8}  {row[4]:>8}  {row[5]:>11}  {row[6]:>16}")


if __name__ == "__main__":
    main()
