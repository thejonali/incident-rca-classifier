from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODELS_DIR, DEFAULT_REPORTS_DIR, FEATURE_PROFILES


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


def main() -> None:
    args = parse_args()
    rows = []
    for model_name in ["gru", "lstm"]:
        for feature_profile in FEATURE_PROFILES:
            metrics = load_profile_metrics(args.models_dir, args.reports_dir, model_name, feature_profile)
            if metrics is None:
                rows.append((model_name.upper(), feature_profile, "missing", "-", "-", "-", "-"))
                continue
            rows.append(
                (
                    model_name.upper(),
                    metrics.get("feature_profile", feature_profile),
                    str(metrics["rows_used"]),
                    f"{metrics['test_accuracy']:.3f}",
                    f"{metrics['test_macro_f1']:.3f}",
                    f"{metrics['test_weighted_f1']:.3f}",
                    f"{metrics['training_seconds']:.1f}s",
                )
            )

        legacy_metrics = load_metrics(args.models_dir, model_name)
        if legacy_metrics is not None:
            rows.append(
                (
                    model_name.upper(),
                    legacy_metrics.get("feature_profile", "legacy"),
                    str(legacy_metrics["rows_used"]),
                    f"{legacy_metrics['test_accuracy']:.3f}",
                    f"{legacy_metrics['test_macro_f1']:.3f}",
                    f"{legacy_metrics['test_weighted_f1']:.3f}",
                    f"{legacy_metrics['training_seconds']:.1f}s",
                )
            )

    print("model  feature_profile  rows  accuracy  macro_f1  weighted_f1  train_time")
    print("-----  ---------------  ----  --------  --------  -----------  ----------")
    for row in rows:
        print(f"{row[0]:<5}  {row[1]:<15}  {row[2]:>4}  {row[3]:>8}  {row[4]:>8}  {row[5]:>11}  {row[6]:>10}")


if __name__ == "__main__":
    main()
