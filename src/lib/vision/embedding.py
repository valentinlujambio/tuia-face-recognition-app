from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from lib.vision.detection import DEFAULT_INSIGHTFACE_ROOT, DEFAULT_MODEL_PACK, resolve_model_pack
from lib.vision.preprocessing import InputNormalization, l2_normalize

DEFAULT_RECOGNIZER_FILE = "w600k_r50.onnx"


class FaceEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Embeddings (N, dim) normalizados L2."""
        ...


class OnnxFaceEmbedder:
    """Extractor de embeddings sobre un modelo ONNX que recibe caras alineadas."""

    def __init__(
        self,
        model_file: str | Path,
        normalization: InputNormalization = InputNormalization(),
        batch_size: int = 64,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        import onnxruntime

        self._session = onnxruntime.InferenceSession(str(model_file), providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name
        self._normalization = normalization
        self._batch_size = batch_size

    @classmethod
    def from_model_pack(
        cls,
        model_pack: str = DEFAULT_MODEL_PACK,
        root: str | Path = DEFAULT_INSIGHTFACE_ROOT,
        **kwargs,
    ) -> "OnnxFaceEmbedder":
        return cls(resolve_model_pack(model_pack, root) / DEFAULT_RECOGNIZER_FILE, **kwargs)

    @property
    def dim(self) -> int:
        return int(self._session.get_outputs()[0].shape[-1])

    def embed(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(faces_bgr) == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        outputs = [
            self._session.run(None, {self._input_name: self._normalization(faces_bgr[i : i + self._batch_size])})[0]
            for i in range(0, len(faces_bgr), self._batch_size)
        ]
        return l2_normalize(np.concatenate(outputs).astype(np.float32))
