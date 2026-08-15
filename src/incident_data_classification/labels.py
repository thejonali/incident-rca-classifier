from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelEncoder:
    label_to_id: dict[str, int]

    @classmethod
    def fit(cls, labels: list[str]) -> "LabelEncoder":
        unique_labels = sorted(set(labels))
        return cls({label: index for index, label in enumerate(unique_labels)})

    @property
    def id_to_label(self) -> dict[int, str]:
        return {index: label for label, index in self.label_to_id.items()}

    def encode(self, labels: list[str]) -> list[int]:
        return [self.label_to_id[label] for label in labels]

    def decode(self, ids: list[int]) -> list[str]:
        lookup = self.id_to_label
        return [lookup[index] for index in ids]

    @property
    def size(self) -> int:
        return len(self.label_to_id)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"label_to_id": self.label_to_id}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelEncoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls({str(label): int(index) for label, index in payload["label_to_id"].items()})

