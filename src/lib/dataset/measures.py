from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

Measure = Callable[[np.ndarray], dict[str, float]]


def exposure(image_bgr: np.ndarray, low: int = 10, high: int = 245) -> dict[str, float]:
    """Brillo medio, contraste (desvío) y fracción de píxeles casi negros o casi blancos, en gris."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "dark_fraction": float((gray <= low).mean()),
        "bright_fraction": float((gray >= high).mean()),
    }


def grayscale(image_bgr: np.ndarray, tolerance: float = 2.0) -> dict[str, float]:
    """is_grayscale = True si los tres canales son prácticamente iguales (foto en B/N guardada como color)."""
    b, g, r = (image_bgr[..., i].astype(np.float32) for i in range(3))
    return {"is_grayscale": float(max(np.abs(b - g).mean(), np.abs(g - r).mean())) < tolerance}


def measure_aligned(df: pd.DataFrame, aligned_root: Path, measures: Sequence[Measure]) -> pd.DataFrame:
    """Aplica cada medida a cada cara alineada. Las filas sin recorte quedan en NaN.

    Las medidas son funciones imagen → dict, así se agregan nuevas sin tocar esta función.
    """
    rows = {}
    with_face = df["aligned_path"].dropna()
    for index, rel_path in tqdm(with_face.items(), total=len(with_face), desc="Midiendo"):
        image = cv2.imread(str(Path(aligned_root) / rel_path))
        rows[index] = {k: v for measure in measures for k, v in measure(image).items()}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(df.index)
