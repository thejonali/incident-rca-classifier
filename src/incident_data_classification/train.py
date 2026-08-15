from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULT_MODELS_DIR, DEFAULT_REPORTS_DIR, RANDOM_SEED, TARGET_COLUMN
from .data import load_incidents, prepare_sequences, split_dataset
from .dataset import resolve_incidents_csv
from .labels import LabelEncoder
from .model import RecurrentIncidentClassifier, get_device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(x: list[list[int]], y: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: RecurrentIncidentClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids)
            loss = criterion(logits, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * input_ids.size(0)
        predictions = logits.argmax(dim=1).detach().cpu().tolist()
        all_predictions.extend(predictions)
        all_targets.extend(targets.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, all_predictions, all_targets


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GRU or LSTM incident classifier")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV path. Defaults to data/raw.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the local CSV before training.")
    parser.add_argument("--model-type", choices=["gru", "lstm"], required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument(
        "--monitor",
        choices=["val_macro_f1", "val_accuracy", "val_loss"],
        default="val_macro_f1",
        help="Validation metric used for best-checkpoint restore and early stopping.",
    )
    parser.add_argument("--prefer-mps", action="store_true", help="Use Apple MPS if available")
    return parser.parse_args()


def is_improvement(metric_name: str, current: float, best: float | None, min_delta: float) -> bool:
    if best is None:
        return True
    if metric_name == "val_loss":
        return current < best - min_delta
    return current > best + min_delta


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_incidents_csv(args.csv, force_download=args.force_download)
    print(f"Using dataset: {csv_path}")

    df = load_incidents(csv_path, max_rows=args.max_rows)
    splits = split_dataset(df)
    tokenizer, sequences = prepare_sequences(splits, vocab_size=args.vocab_size, max_length=args.max_length)
    label_encoder = LabelEncoder.fit(splits.y_train)

    y_train = label_encoder.encode(splits.y_train)
    y_val = label_encoder.encode(splits.y_val)
    y_test = label_encoder.encode(splits.y_test)

    train_loader = make_loader(sequences["train"], y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(sequences["val"], y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(sequences["test"], y_test, args.batch_size, shuffle=False)

    device = get_device(prefer_mps=args.prefer_mps)
    model = RecurrentIncidentClassifier(
        model_type=args.model_type,
        vocab_size=tokenizer.size,
        num_classes=label_encoder.size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_metric: float | None = None
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pred, train_true = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_pred, val_true = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": accuracy_score(train_true, train_pred),
            "train_macro_f1": f1_score(train_true, train_pred, average="macro"),
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(val_true, val_pred),
            "val_macro_f1": f1_score(val_true, val_pred, average="macro"),
        }
        history.append(row)
        print(
            f"{args.model_type.upper()} epoch {epoch}/{args.epochs} "
            f"train_acc={row['train_accuracy']:.3f} val_acc={row['val_accuracy']:.3f} "
            f"val_macro_f1={row['val_macro_f1']:.3f}"
        )

        current_metric = float(row[args.monitor])
        if is_improvement(args.monitor, current_metric, best_metric, args.min_delta):
            best_metric = current_metric
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after epoch {epoch}; best {args.monitor}="
                    f"{best_metric:.3f} at epoch {best_epoch}."
                )
                break

    training_seconds = time.perf_counter() - start
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_pred, test_true = run_epoch(model, test_loader, criterion, device)
    labels = [label for label, _ in sorted(label_encoder.label_to_id.items(), key=lambda item: item[1])]
    metrics = {
        "model_type": args.model_type,
        "rows_used": int(len(df)),
        "classes": labels,
        "class_distribution": df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "device": str(device),
        "training_seconds": training_seconds,
        "best_epoch": best_epoch,
        "best_monitor": args.monitor,
        "best_monitor_value": best_metric,
        "test_loss": test_loss,
        "test_accuracy": accuracy_score(test_true, test_pred),
        "test_macro_f1": f1_score(test_true, test_pred, average="macro"),
        "test_weighted_f1": f1_score(test_true, test_pred, average="weighted"),
        "history": history,
        "classification_report": classification_report(test_true, test_pred, target_names=labels, zero_division=0),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer.size,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "monitor": args.monitor,
            "max_rows": args.max_rows,
        },
    }

    artifact_dir = args.models_dir / args.model_type
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_type": args.model_type,
            "vocab_size": tokenizer.size,
            "num_classes": label_encoder.size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
        },
        artifact_dir / "model.pt",
    )
    tokenizer.save(artifact_dir / "tokenizer.json")
    label_encoder.save(artifact_dir / "label_encoder.json")
    save_json(artifact_dir / "metrics.json", metrics)
    save_json(args.reports_dir / f"{args.model_type}_metrics.json", metrics)

    reference_columns = [
        "incident_id",
        "input_text",
        "root_cause_category",
        "remediation_that_worked",
        "prevention_recommendation",
    ]
    df[reference_columns].to_json(artifact_dir / "reference_incidents.json", orient="records", indent=2)

    print(f"\nSaved {args.model_type.upper()} artifacts to {artifact_dir}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Training time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()
