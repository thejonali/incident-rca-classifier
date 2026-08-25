from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    FEATURE_PROFILES,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from .data import get_feature_columns, load_incidents, split_dataset
from .dataset import resolve_incidents_csv
from .labels import LabelEncoder
from .model import get_device
from .train import save_json
from .transformer_utils import TransformerIncidentDataset, directory_size_mb, resolve_transformer_artifact_dir


DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small transformer incident classifier")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before training.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_TRANSFORMER_MODEL)
    parser.add_argument("--artifact-name", type=str, default="distilbert")
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default=DEFAULT_FEATURE_PROFILE,
        help="Input field profile used to build incident text.",
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Cap optimizer steps for CPU-friendly benchmark runs. Use 0 for full epochs.",
    )
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def make_loader(
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TransformerIncidentDataset(texts, labels, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item()) * batch["labels"].size(0)
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())

    return total_loss / max(1, len(loader.dataset)), predictions, targets


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    print(f"Using dataset: {csv_path}")
    print(f"Using transformer model: {args.model_name}")
    print(f"Using feature profile: {args.feature_profile}")

    df = load_incidents(csv_path, max_rows=args.max_rows, feature_profile=args.feature_profile)
    splits = split_dataset(df)
    label_encoder = LabelEncoder.fit(splits.y_train)
    y_train = label_encoder.encode(splits.y_train)
    y_val = label_encoder.encode(splits.y_val)
    y_test = label_encoder.encode(splits.y_test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = make_loader(splits.x_train, y_train, tokenizer, args.max_length, args.batch_size, shuffle=True)
    val_loader = make_loader(splits.x_val, y_val, tokenizer, args.max_length, args.batch_size, shuffle=False)
    test_loader = make_loader(splits.x_test, y_test, tokenizer, args.max_length, args.batch_size, shuffle=False)

    device = get_device(prefer_mps=args.prefer_mps)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=label_encoder.size,
        id2label={index: label for label, index in label_encoder.label_to_id.items()},
        label2id=label_encoder.label_to_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    full_training_steps = args.epochs * len(train_loader)
    total_steps = min(full_training_steps, args.max_steps) if args.max_steps > 0 else full_training_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, total_steps))

    history: list[dict[str, float]] = []
    steps_completed = 0
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_predictions: list[int] = []
        train_targets: list[int] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            steps_completed += 1
            total_loss += float(outputs.loss.item()) * batch["labels"].size(0)
            train_predictions.extend(outputs.logits.argmax(dim=1).detach().cpu().tolist())
            train_targets.extend(batch["labels"].detach().cpu().tolist())
            if args.max_steps > 0 and steps_completed >= args.max_steps:
                break

        val_loss, val_predictions, val_targets = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "steps_completed": steps_completed,
                "train_loss": total_loss / max(1, len(train_targets)),
                "train_accuracy": accuracy_score(train_targets, train_predictions) if train_targets else 0.0,
                "train_macro_f1": f1_score(train_targets, train_predictions, average="macro") if train_targets else 0.0,
                "val_loss": val_loss,
                "val_accuracy": accuracy_score(val_targets, val_predictions),
                "val_macro_f1": f1_score(val_targets, val_predictions, average="macro"),
            }
        )
        print(
            f"Transformer epoch {epoch}/{args.epochs} steps={steps_completed} "
            f"val_acc={history[-1]['val_accuracy']:.3f} val_macro_f1={history[-1]['val_macro_f1']:.3f}"
        )
        if args.max_steps > 0 and steps_completed >= args.max_steps:
            break

    training_seconds = time.perf_counter() - training_start

    inference_start = time.perf_counter()
    test_loss, test_predictions, test_targets = evaluate_model(model, test_loader, device)
    inference_seconds = time.perf_counter() - inference_start
    inference_latency_ms = (inference_seconds / max(1, len(test_targets))) * 1000
    labels = [label for label, _ in sorted(label_encoder.label_to_id.items(), key=lambda item: item[1])]

    artifact_dir = resolve_transformer_artifact_dir(args.models_dir, args.artifact_name, args.feature_profile)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
    label_encoder.save(artifact_dir / "label_encoder.json")

    metrics = {
        "model_type": args.artifact_name,
        "model_family": "transformer",
        "pretrained_model_name": args.model_name,
        "feature_profile": args.feature_profile,
        "feature_columns": get_feature_columns(args.feature_profile),
        "rows_used": int(len(df)),
        "classes": labels,
        "class_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "device": str(device),
        "training_seconds": training_seconds,
        "inference_latency_ms": inference_latency_ms,
        "model_size_mb": directory_size_mb(artifact_dir),
        "steps_completed": steps_completed,
        "test_loss": test_loss,
        "test_accuracy": accuracy_score(test_targets, test_predictions),
        "test_macro_f1": f1_score(test_targets, test_predictions, average="macro"),
        "test_weighted_f1": f1_score(test_targets, test_predictions, average="weighted"),
        "classification_report": classification_report(
            test_targets,
            test_predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "history": history,
        "hyperparameters": {
            "max_rows": args.max_rows,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
        },
    }
    save_json(artifact_dir / "metrics.json", metrics)
    save_json(args.reports_dir / f"transformer_{args.artifact_name}_{args.feature_profile}_metrics.json", metrics)

    print(f"\nSaved transformer artifacts to {artifact_dir}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Inference latency: {inference_latency_ms:.3f} ms/incident")
    print(f"Model size: {metrics['model_size_mb']:.1f} MB")
    print(f"Training time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()
