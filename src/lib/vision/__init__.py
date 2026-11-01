"""Pipeline facial compartido entre las notebooks y el backend (face_service.py).

Detectar, alinear, normalizar y extraer el embedding. Tenerlo en un solo módulo evita
que entrenamiento e inferencia preprocesen distinto.
"""

from lib.vision.alignment import FaceAligner
from lib.vision.detection import FaceDetector, ScrfdDetector, clip_box, resolve_model_pack
from lib.vision.embedding import FaceEmbedder, OnnxFaceEmbedder
from lib.vision.preprocessing import InputNormalization, l2_normalize
from lib.vision.types import Detection

__all__ = [
    "Detection",
    "FaceAligner",
    "FaceDetector",
    "FaceEmbedder",
    "InputNormalization",
    "OnnxFaceEmbedder",
    "ScrfdDetector",
    "clip_box",
    "l2_normalize",
    "resolve_model_pack",
]
