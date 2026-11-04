from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from lib.vision.types import Detection

DEFAULT_MODEL_PACK = "buffalo_l"
DEFAULT_DETECTOR_FILE = "det_10g.onnx"
DEFAULT_INSIGHTFACE_ROOT = "~/.insightface"


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Devuelve las caras ordenadas de mayor a menor área."""
        ...


def clip_box(
    box: Sequence[float], height: int, width: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def resolve_model_pack(
    model_pack: str = DEFAULT_MODEL_PACK, root: str | Path = DEFAULT_INSIGHTFACE_ROOT
) -> Path:
    """Descarga (si hace falta) el paquete de modelos de InsightFace y devuelve su carpeta."""
    from insightface.utils.storage import ensure_available

    return Path(ensure_available("models", model_pack, root=str(Path(root).expanduser())))


class ScrfdDetector:
    """Detector SCRFD de InsightFace (det_10g.onnx del paquete buffalo_l).

    Una sola pasada devuelve caja, score y los 5 landmarks que necesita la alineación.

    pad_ratio agrega un borde negro de pad_ratio × el lado mayor antes de detectar.
    Sirve para imágenes ya recortadas a la cara: si el rostro ocupa todo el cuadro,
    SCRFD no lo encuentra. Las coordenadas se devuelven en la imagen original.
    """

    def __init__(
        self,
        model_file: str | Path,
        score_threshold: float = 0.5,
        input_size: int = 640,
        min_face_size: int = 20,
        pad_ratio: float = 0.0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        from insightface.model_zoo import get_model

        self.score_threshold = score_threshold
        self.min_face_size = min_face_size
        self.pad_ratio = pad_ratio
        self._model = get_model(str(model_file), providers=list(providers))
        if self._model is None:
            raise ValueError(f"Could not load detector: {model_file}")
        self._model.prepare(
            ctx_id=-1,
            input_size=(input_size, input_size),
            det_thresh=score_threshold,
        )

    @classmethod
    def from_model_pack(
        cls,
        model_pack: str = DEFAULT_MODEL_PACK,
        root: str | Path = DEFAULT_INSIGHTFACE_ROOT,
        **kwargs,
    ) -> "ScrfdDetector":
        return cls(resolve_model_pack(model_pack, root) / DEFAULT_DETECTOR_FILE, **kwargs)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        height, width = image_bgr.shape[:2]
        pad = int(round(self.pad_ratio * max(height, width)))
        padded = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) if pad else image_bgr
        boxes, kpss = self._model.detect(padded)
        detections = []
        for box, kps in zip(boxes, kpss):
            detection = Detection(
                bbox=clip_box(box[:4] - pad, height, width),
                score=float(box[4]),
                keypoints=np.asarray(kps, dtype=np.float32) - pad,
            )
            if detection.score >= self.score_threshold and detection.min_side >= self.min_face_size:
                detections.append(detection)
        return sorted(detections, key=lambda d: d.area, reverse=True)
