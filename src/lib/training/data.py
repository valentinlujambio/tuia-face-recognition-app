"""Datos para entrenar el Modelo A: caras alineadas en RAM y Dataset de PyTorch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.vision import InputNormalization

Augment = Callable[..., dict]


class LabelEncoder:
    """Identidad (str) <-> índice de clase (int). Se ajusta solo con las identidades de train."""

    def __init__(self, identities: Sequence[str]) -> None:
        self.classes: list[str] = sorted(set(identities))
        self._index = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def encode(self, identities: Sequence[str]) -> np.ndarray:
        return np.array([self._index[name] for name in identities], dtype=np.int64)


def load_faces(root: str | Path, relative_paths: Sequence[str], workers: int = 16) -> np.ndarray:
    """Lee las caras alineadas (BGR uint8) en paralelo y las apila en un array (N, H, W, 3)."""
    root = Path(root)

    def read(relative: str) -> np.ndarray:
        image = cv2.imread(str(root / relative))
        if image is None:
            raise FileNotFoundError(root / relative)
        return image

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(read, relative_paths)))


class FaceDataset(Dataset):
    """Caras precargadas -> (tensor CHW normalizado igual que en la app, etiqueta)."""

    def __init__(
        self,
        faces_bgr: np.ndarray,
        labels: np.ndarray,
        normalization: InputNormalization = InputNormalization(),
        augment: Augment | None = None,
    ) -> None:
        if len(faces_bgr) != len(labels):
            raise ValueError("faces_bgr y labels deben tener la misma cantidad de filas")
        self.faces = faces_bgr
        self.labels = np.asarray(labels, dtype=np.int64)
        self.normalization = normalization
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        face = self.faces[index]
        if self.augment is not None:
            face = self.augment(image=face)["image"]
        return torch.from_numpy(self.normalization([face])[0]), int(self.labels[index])
