from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.schemas import EmbeddingRecord
from lib.services.face_service import FaceService

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "docs"
RECOGNIZER_FILE = "w600k_r50.onnx"
DETECTOR_FILE = "det_10g.onnx"
# Valores de .env.docker.example
FACE_SIZE = 112
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.334


class InMemoryStore:
    """Store en memoria con la interfaz que usa FaceService (all / append)."""

    def __init__(self) -> None:
        self.records: list[EmbeddingRecord] = []

    def all(self) -> list[EmbeddingRecord]:
        return list(self.records)

    def append(self, record: EmbeddingRecord) -> None:
        self.records.append(record)


def _find_model(name: str) -> Path | None:
    for folder in (ROOT / "models", Path.home() / ".insightface" / "models" / "buffalo_l"):
        if (folder / name).exists():
            return folder / name
    return None


@pytest.fixture(scope="session")
def _service() -> FaceService:
    recognizer, detector = _find_model(RECOGNIZER_FILE), _find_model(DETECTOR_FILE)
    if recognizer is None or detector is None:
        pytest.skip("Faltan w600k_r50.onnx y det_10g.onnx en models/ (ver 'Cómo correrlo' en el README)")
    return FaceService(
        store=InMemoryStore(),
        similarity_metric="cosine",
        similarity_threshold=SIMILARITY_THRESHOLD,
        face_size=FACE_SIZE,
        model_path=recognizer,
        detector_path=detector,
    )


@pytest.fixture
def service(_service: FaceService, tmp_path: Path) -> FaceService:
    """El modelo y el detector se cargan una sola vez; el store y la carpeta de salida son de cada test."""
    _service.store = InMemoryStore()
    _service.output_path = tmp_path
    return _service


def _photo(relpath: str) -> Path:
    path = PHOTOS / relpath
    if not path.exists():
        pytest.skip(f"Falta la foto de prueba {path}")
    return path


@pytest.fixture
def valentin_photos() -> list[Path]:
    return [_photo(f"fotos_valentin/val_{i}.jpeg") for i in (1, 2, 3, 4)]


@pytest.fixture
def lucas_photos() -> list[Path]:
    return [_photo("fotos_lucas/lucas_1.jpeg"), _photo("fotos_lucas/lucas_2.jpeg")]


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, f"No se pudo leer {path}"
    return image


@pytest.fixture
def two_faces_image(valentin_photos, lucas_photos) -> np.ndarray:
    """Dos personas lado a lado, a la misma altura."""
    height = 480
    resized = []
    for path in (valentin_photos[0], lucas_photos[0]):
        image = read_bgr(path)
        resized.append(cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height)))
    return np.hstack(resized)


@pytest.fixture
def noise_image() -> np.ndarray:
    """Imagen sin rostros."""
    return np.random.RandomState(0).randint(0, 255, (400, 600, 3), dtype=np.uint8)
