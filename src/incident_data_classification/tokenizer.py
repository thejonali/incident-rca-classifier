from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class TextTokenizer:
    vocab_size: int = 8000
    max_length: int = 96
    token_to_id: dict[str, int] | None = None

    def fit(self, texts: list[str]) -> None:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(TOKEN_RE.findall(text.lower()))

        most_common = counts.most_common(max(0, self.vocab_size - 2))
        self.token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.token_to_id.update({token: index + 2 for index, (token, _) in enumerate(most_common)})

    def encode(self, text: str) -> list[int]:
        if self.token_to_id is None:
            raise ValueError("Tokenizer has not been fit")

        tokens = TOKEN_RE.findall(text.lower())
        ids = [self.token_to_id.get(token, self.token_to_id[UNK_TOKEN]) for token in tokens[: self.max_length]]
        if len(ids) < self.max_length:
            ids.extend([self.token_to_id[PAD_TOKEN]] * (self.max_length - len(ids)))
        return ids

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    @property
    def size(self) -> int:
        if self.token_to_id is None:
            raise ValueError("Tokenizer has not been fit")
        return len(self.token_to_id)

    def save(self, path: Path) -> None:
        if self.token_to_id is None:
            raise ValueError("Tokenizer has not been fit")
        payload = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TextTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocab_size=int(payload["vocab_size"]),
            max_length=int(payload["max_length"]),
            token_to_id={str(key): int(value) for key, value in payload["token_to_id"].items()},
        )

