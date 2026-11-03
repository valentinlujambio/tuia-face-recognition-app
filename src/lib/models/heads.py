"""Cabezas de entrenamiento: convierten embeddings en logits para la pérdida.

Se usan solo durante el entrenamiento y no se exportan. Todas comparten la interfaz
`forward(embeddings, labels) -> logits`, así el loop de entrenamiento no depende de cuál se usa
y se puede agregar una de margen angular (CosFace / ArcFace) sin tocarlo.
"""

from __future__ import annotations

import torch
from torch import nn


class SoftmaxHead(nn.Module):
    """Clasificador lineal sobre las identidades de train (enfoque clasificación + embedding)."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(embeddings)
