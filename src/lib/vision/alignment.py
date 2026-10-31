from __future__ import annotations

import numpy as np


class FaceAligner:
    """Alinea con una transformación de similitud que lleva los 5 landmarks
    a la plantilla de ArcFace y recorta a face_size × face_size."""

    def __init__(self, face_size: int = 112) -> None:
        if face_size % 112 != 0 and face_size % 128 != 0:
            raise ValueError("face_size must be a multiple of 112 or 128 (ArcFace template).")
        self.face_size = face_size

    def align(self, image_bgr: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        from insightface.utils.face_align import norm_crop

        return norm_crop(image_bgr, np.asarray(keypoints, dtype=np.float32), image_size=self.face_size)
