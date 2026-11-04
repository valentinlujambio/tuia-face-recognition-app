from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Rostro detectado en la imagen original.

    bbox: (x1, y1, x2, y2) en píxeles enteros.
    keypoints: (5, 2) en coordenadas absolutas de la imagen, orden InsightFace:
        ojo izquierdo, ojo derecho, nariz, comisura izquierda, comisura derecha.
    """

    bbox: tuple[int, int, int, int]
    score: float
    keypoints: np.ndarray

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    @property
    def area(self) -> int:
        return self.width * self.height

    def keypoints_relative_to_box(self) -> np.ndarray:
        """Landmarks relativos a la esquina superior izquierda de la caja (lo que espera el frontend)."""
        return self.keypoints - np.array(self.bbox[:2], dtype=self.keypoints.dtype)
