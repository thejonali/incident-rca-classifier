from __future__ import annotations

import torch
from torch import nn


class RecurrentIncidentClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if model_type not in {"gru", "lstm"}:
            raise ValueError("model_type must be 'gru' or 'lstm'")

        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.recurrent = recurrent_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, hidden = self.recurrent(embedded)
        if isinstance(hidden, tuple):
            hidden_state = hidden[0][-1]
        else:
            hidden_state = hidden[-1]
        return self.classifier(self.dropout(hidden_state))


def get_device(prefer_mps: bool = False) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

