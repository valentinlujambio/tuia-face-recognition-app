"""Modelo A: CNN residual chica que mapea una cara alineada de 112x112 a un embedding.

Solo el extractor de embeddings se exporta (.onnx / .pth). La cabeza de clasificación que se usa
para entrenar vive en `lib.models.heads` y se descarta después del entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class CustomCNNConfig:
    """Hiperparámetros de arquitectura.

    channels: canales de cada etapa; cada etapa reduce la resolución a la mitad (112 -> 7 con 4 etapas).
    blocks_per_stage: bloques residuales por etapa (el primero hace el stride 2).
    """

    channels: tuple[int, ...] = (32, 64, 128, 256)
    blocks_per_stage: int = 2
    stem_channels: int = 32
    embedding_dim: int = 128
    dropout: float = 0.2
    input_size: int = 112

    def __post_init__(self) -> None:
        if not self.channels or self.blocks_per_stage < 1:
            raise ValueError("channels no puede estar vacío y blocks_per_stage debe ser >= 1")
        if self.input_size % (2 ** len(self.channels)):
            raise ValueError(f"input_size={self.input_size} no es divisible por 2^{len(self.channels)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")


def conv_bn(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class ResidualBlock(nn.Module):
    """Bloque básico de ResNet (dos conv 3x3) con atajo 1x1 cuando cambia la forma."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_bn(in_ch, out_ch, stride=stride), nn.PReLU(out_ch), conv_bn(out_ch, out_ch))
        self.shortcut = (
            nn.Identity() if stride == 1 and in_ch == out_ch else conv_bn(in_ch, out_ch, kernel=1, stride=stride)
        )
        self.act = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class CustomFaceCNN(nn.Module):
    """Extractor de embeddings: (N, 3, 112, 112) normalizado con InputNormalization -> (N, embedding_dim).

    El embedding se devuelve sin normalizar; la normalización L2 la hace quien lo consume
    (OnnxFaceEmbedder / FaceService), igual que con el modelo preentrenado.
    """

    def __init__(self, config: CustomCNNConfig = CustomCNNConfig()) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(conv_bn(3, config.stem_channels), nn.PReLU(config.stem_channels))
        self.stages = nn.Sequential(*self._build_stages(config))
        self.embedding = nn.Sequential(
            nn.BatchNorm2d(config.channels[-1]),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.channels[-1], config.embedding_dim, bias=False),
            nn.BatchNorm1d(config.embedding_dim),
        )

    @staticmethod
    def _build_stages(config: CustomCNNConfig) -> list[nn.Module]:
        stages, in_ch = [], config.stem_channels
        for out_ch in config.channels:
            blocks = [ResidualBlock(in_ch, out_ch, stride=2)]
            blocks += [ResidualBlock(out_ch, out_ch) for _ in range(config.blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        return stages

    @classmethod
    def from_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "CustomFaceCNN":
        """Reconstruye el extractor desde un `best.pt` de CheckpointStore (state_dict + config)."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(checkpoint["config"])
        config["channels"] = tuple(config["channels"])
        model = cls(CustomCNNConfig(**config))
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.stages(self.stem(x)))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
