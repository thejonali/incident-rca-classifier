from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved GRU and LSTM metrics")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def load_metrics(models_dir: Path, model_name: str) -> dict | None:
    metrics_path = models_dir / model_name / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = []
    for model_name in ["gru", "lstm"]:
        metrics = load_metrics(args.models_dir, model_name)
        if metrics is None:
            rows.append((model_name.upper(), "missing", "-", "-", "-", "-"))
            continue
        rows.append(
            (
                model_name.upper(),
                str(metrics["rows_used"]),
                f"{metrics['test_accuracy']:.3f}",
                f"{metrics['test_macro_f1']:.3f}",
                f"{metrics['test_weighted_f1']:.3f}",
                f"{metrics['training_seconds']:.1f}s",
            )
        )

    print("model  rows  accuracy  macro_f1  weighted_f1  train_time")
    print("-----  ----  --------  --------  -----------  ----------")
    for row in rows:
        print(f"{row[0]:<5}  {row[1]:>4}  {row[2]:>8}  {row[3]:>8}  {row[4]:>11}  {row[5]:>10}")


if __name__ == "__main__":
    main()

