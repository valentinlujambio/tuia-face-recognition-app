from __future__ import annotations

import copy
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd

from lib.vision import FaceEmbedder


def build_train_augmentation(seed: int | None = None) -> A.Compose:
    """Augmentation para caras ya alineadas (BGR uint8, face_size x face_size).

    Solo variaciones que pueden ocurrir en una foto real y no cambian la identidad.
    Se aplica únicamente sobre train; la normalización no es augmentation y vive en
    lib.vision.InputNormalization. Las imágenes llegan en BGR, como las lee cv2: ToGray y
    HueSaturationValue suponen RGB, pero con estos rangos la diferencia es despreciable.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-10, 10), translate_percent=(-0.05, 0.05), scale=(0.95, 1.05),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=0, p=0.3),
            A.ToGray(p=0.1),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.15, 0.3),
                hole_width_range=(0.15, 0.3),
                fill=0,
                p=0.3,
            ),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ],
        seed=seed,
    )


def identity_preservation(
    embedder: FaceEmbedder, faces_bgr: Sequence[np.ndarray], transform: A.Compose, versions: int = 1
) -> np.ndarray:
    """Coseno entre el embedding de cada cara y el de sus versiones aumentadas (versions por cara).

    Valores altos indican que la transformación no cambia quién es la persona para el modelo.
    """
    original = embedder.embed(faces_bgr)
    sims = []
    for _ in range(versions):
        augmented = embedder.embed([transform(image=face)["image"] for face in faces_bgr])
        sims.append(np.einsum("ij,ij->i", original, augmented))
    return np.concatenate(sims)


def transform_effects(
    embedder: FaceEmbedder, faces_bgr: Sequence[np.ndarray], augment: A.Compose, seed: int | None = None
) -> pd.DataFrame:
    """identity_preservation de cada transformación del pipeline aplicada sola y siempre (p = 1)."""
    rows = []
    for transform in augment.transforms:
        single = copy.deepcopy(transform)
        single.p = 1.0
        sims = identity_preservation(embedder, faces_bgr, A.Compose([single], seed=seed))
        rows.append({"transformación": type(transform).__name__, "p en el pipeline": transform.p,
                     "mediana": np.median(sims), "p5": np.percentile(sims, 5), "mínimo": sims.min()})
    return pd.DataFrame(rows).set_index("transformación")
