from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class InputNormalization:
    """Convierte caras alineadas (BGR uint8, HWC) en el tensor de entrada del modelo (NCHW float32).

    Los valores por defecto son los de ArcFace de InsightFace: RGB y (x - 127.5) / 127.5.
    Es el único lugar donde se decide el orden de canales y la escala de píxeles.
    """

    mean: float = 127.5
    std: float = 127.5
    to_rgb: bool = True

    def __call__(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        batch = np.stack(
            [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if self.to_rgb else face for face in faces_bgr]
        ).astype(np.float32)
        batch = (batch - self.mean) / self.std
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)
