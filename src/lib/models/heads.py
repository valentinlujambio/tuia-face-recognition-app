"""Cabezas de entrenamiento: convierten embeddings en logits para la pérdida.

Se usan solo durante el entrenamiento y no se exportan. Todas comparten la interfaz
`forward(embeddings, labels) -> logits`, así el loop de entrenamiento no depende de cuál se usa
y se puede agregar una de margen angular (CosFace / ArcFace) sin tocarlo.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxHead(nn.Module):
    """Clasificador lineal sobre las identidades de train (enfoque clasificación + embedding)."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(embeddings)


class ArcFaceHead(nn.Module):
    """Margen angular aditivo (ArcFace, Deng et al. 2019): logits = s * cos(theta + m) para la clase correcta.

    Embeddings y pesos se normalizan L2, así el logit depende solo del ángulo. Sumar m al ángulo de la
    clase correcta obliga a que cada identidad quede más compacta y más lejos de las otras, que es lo
    que mide la verificación por coseno. Sin labels (evaluación) devuelve s * cos(theta) sin margen.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.threshold = math.cos(math.pi - margin)  # más allá, theta + m pasaría de pi
        self.fallback = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        # En float32 aunque el resto corra con AMP: sqrt(1 - cos^2) cerca de 1 es inestable en float16.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            return self._logits(embeddings.float(), labels)

    def _logits(self, embeddings: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None or not self.training:
            return self.scale * cosine
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))
        cos_theta_m = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        cos_theta_m = torch.where(cosine > self.threshold, cos_theta_m, cosine - self.fallback)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).bool()
        return self.scale * torch.where(one_hot, cos_theta_m, cosine)

    @torch.no_grad()
    def init_from_centroids(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Inicializa el vector de cada clase con el centroide normalizado de sus embeddings.

        Para fine-tuning desde un extractor ya entrenado: la cabeza arranca alineada con las
        identidades en vez de al azar, así el margen no impide que la clase correcta gane al inicio.
        """
        embeddings = F.normalize(embeddings.float(), dim=1)
        for c in range(self.weight.size(0)):
            members = embeddings[labels == c]
            if len(members) == 0:
                raise ValueError(f"la clase {c} no tiene embeddings")
            self.weight[c] = F.normalize(members.mean(0), dim=0).to(self.weight)
