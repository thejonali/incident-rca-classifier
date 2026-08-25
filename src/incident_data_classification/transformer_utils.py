from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import LabelEncoder


class TransformerIncidentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def resolve_transformer_artifact_dir(models_dir: Path, model_name: str, feature_profile: str) -> Path:
    return models_dir / "transformers" / model_name / feature_profile


def load_transformer_artifacts(
    artifact_dir: Path,
    device: torch.device,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, LabelEncoder]:
    if not (artifact_dir / "label_encoder.json").exists():
        raise FileNotFoundError(f"Transformer artifact not found at {artifact_dir}. Train it first.")

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    label_encoder = LabelEncoder.load(artifact_dir / "label_encoder.json")
    model.to(device)
    model.eval()
    return model, tokenizer, label_encoder


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total_bytes = sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())
    return total_bytes / (1024 * 1024)
