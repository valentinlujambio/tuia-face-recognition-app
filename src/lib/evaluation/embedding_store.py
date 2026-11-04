"""Contrato entre las notebooks de cada modelo (2a, 2b) y la evaluación (2-train).

Cada modelo guarda sus embeddings en `results/<modelo>/embeddings.npz` con `save_embeddings`
y la evaluación los lee con `load_embeddings`, sin saber qué modelo los generó.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val_closed", "test_closed", "val_open", "test_open")
_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class EmbeddingSet:
    """Embeddings de un modelo, alineados fila a fila con `splits/splits.csv`.

    paths: `aligned_path` de cada cara. labels: identidad. splits: valor de la columna `split`.
    embeddings: float32 (N, D) normalizados L2.
    """

    model: str
    paths: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model no puede estar vacío")
        if self.embeddings.ndim != 2 or self.embeddings.dtype != np.float32:
            raise ValueError(f"embeddings debe ser float32 (N, D), no {self.embeddings.dtype} {self.embeddings.shape}")
        n = len(self.embeddings)
        if not (len(self.paths) == len(self.labels) == len(self.splits) == n):
            raise ValueError("paths, labels, splits y embeddings deben tener la misma cantidad de filas")
        unknown = set(np.unique(self.splits)) - set(SPLITS)
        if unknown:
            raise ValueError(f"splits desconocidos: {sorted(unknown)}")
        norms = np.linalg.norm(self.embeddings, axis=1)
        if n and not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise ValueError("los embeddings tienen que estar normalizados L2")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def subset(self, *splits: str) -> "EmbeddingSet":
        mask = np.isin(self.splits, splits)
        return EmbeddingSet(self.model, self.paths[mask], self.labels[mask], self.splits[mask], self.embeddings[mask])


def results_path(root: str | Path, model: str) -> Path:
    return Path(root) / "results" / model / "embeddings.npz"


def save_embeddings(embedding_set: EmbeddingSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model=np.array(embedding_set.model),
        paths=embedding_set.paths.astype(str),
        labels=embedding_set.labels.astype(str),
        splits=embedding_set.splits.astype(str),
        embeddings=embedding_set.embeddings,
    )
    return path


def load_embeddings(path: str | Path) -> EmbeddingSet:
    with np.load(path, allow_pickle=False) as data:
        return EmbeddingSet(
            model=str(data["model"]),
            paths=data["paths"],
            labels=data["labels"],
            splits=data["splits"],
            embeddings=data["embeddings"],
        )
