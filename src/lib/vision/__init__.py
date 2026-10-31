"""Pipeline facial compartido entre las notebooks y el backend (face_service.py).

Tenerlo en un solo módulo evita que entrenamiento e inferencia preprocesen distinto.
"""

from lib.vision.alignment import FaceAligner
from lib.vision.detection import FaceDetector, ScrfdDetector, clip_box, resolve_model_pack
from lib.vision.types import Detection

__all__ = [
    "Detection",
    "FaceAligner",
    "FaceDetector",
    "ScrfdDetector",
    "clip_box",
    "resolve_model_pack",
]
