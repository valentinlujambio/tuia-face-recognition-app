"""Costo computacional de un extractor de embeddings: tamaño, parámetros y tiempo por rostro en CPU."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from lib.vision import FaceEmbedder


def onnx_parameters(model_file: str | Path) -> int:
    """Cantidad de parámetros de un modelo ONNX (suma de los tensores de pesos)."""
    import onnx

    model = onnx.load(str(model_file), load_external_data=False)
    return int(sum(np.prod(tensor.dims) for tensor in model.graph.initializer))


def file_size_mb(model_file: str | Path) -> float:
    return Path(model_file).stat().st_size / 2**20


def ms_per_face(embedder: FaceEmbedder, faces_bgr: Sequence[np.ndarray], batch_size: int = 1, repeats: int = 3) -> float:
    """Mediana de milisegundos por rostro.

    batch_size=1 es el caso de la app (una cara por llamada); se descarta una pasada de calentamiento.
    """
    embedder.embed(faces_bgr[:batch_size])
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(0, len(faces_bgr), batch_size):
            embedder.embed(faces_bgr[i : i + batch_size])
        times.append((time.perf_counter() - start) / len(faces_bgr) * 1000)
    return float(np.median(times))
