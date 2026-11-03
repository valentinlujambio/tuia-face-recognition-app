import json
from pathlib import Path

import numpy as np
import pytest

from conftest import EMBEDDING_DIM, FACE_SIZE, SIMILARITY_THRESHOLD, read_bgr
from lib.schemas import AlignedFace, EmbeddingRecord
from lib.services.face_service import FaceService


def _cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embedding_of(service: FaceService, path) -> list[float]:
    image = read_bgr(path)
    (box,) = service.detect_faces(image)
    return service.extract_embedding_from_face(service.align_face(image, box))


# --- _clip_xyxy (no necesita modelos) ---------------------------------------------------------

class TestClipXyxy:
    def test_box_inside_is_unchanged(self):
        assert FaceService._clip_xyxy(10, 20, 50, 60, height=100, width=100) == (10, 20, 50, 60)

    def test_box_outside_is_clipped_to_image(self):
        assert FaceService._clip_xyxy(-5, -5, 120, 130, height=100, width=100) == (0, 0, 100, 100)

    def test_degenerate_box_keeps_at_least_one_pixel(self):
        x1, y1, x2, y2 = FaceService._clip_xyxy(50, 50, 50, 40, height=100, width=100)
        assert x2 > x1 and y2 > y1


# --- detect_faces -----------------------------------------------------------------------------

class TestDetectFaces:
    def test_one_face(self, service, valentin_photos):
        image = read_bgr(valentin_photos[0])
        boxes = service.detect_faces(image)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]
        assert all(isinstance(v, int) for v in boxes[0])
        height, width = image.shape[:2]
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height

    def test_two_faces(self, service, two_faces_image):
        assert len(service.detect_faces(two_faces_image)) == 2

    def test_no_faces_returns_empty_list(self, service, noise_image):
        # Una lista vacía, no None ni una excepción (enunciado 4.2)
        assert service.detect_faces(noise_image) == []


# --- align_face -------------------------------------------------------------------------------

class TestAlignFace:
    def test_aligned_image_and_keypoints(self, service, valentin_photos):
        image = read_bgr(valentin_photos[0])
        (box,) = service.detect_faces(image)
        aligned = service.align_face(image, box)

        assert isinstance(aligned, AlignedFace)
        assert tuple(aligned.bbox) == box
        assert aligned.image.shape == (FACE_SIZE, FACE_SIZE, 3)
        assert aligned.image.dtype == np.uint8
        assert np.asarray(aligned.keypoints).shape == (5, 2)

    def test_keypoints_are_relative_to_the_box(self, service, valentin_photos):
        # El frontend les suma (x1, y1): si fueran absolutos quedarían desplazados
        image = read_bgr(valentin_photos[0])
        (box,) = service.detect_faces(image)
        keypoints = np.asarray(service.align_face(image, box).keypoints)
        width, height = box[2] - box[0], box[3] - box[1]
        assert (keypoints[:, 0] > -0.2 * width).all() and (keypoints[:, 0] < 1.2 * width).all()
        assert (keypoints[:, 1] > -0.2 * height).all() and (keypoints[:, 1] < 1.2 * height).all()

    def test_keypoints_follow_insightface_order(self, service, valentin_photos):
        # k0 ojo izquierdo, k1 ojo derecho, k2 nariz, k3 y k4 comisuras
        image = read_bgr(valentin_photos[0])
        (box,) = service.detect_faces(image)
        k = np.asarray(service.align_face(image, box).keypoints)
        assert k[0, 0] < k[1, 0] and k[3, 0] < k[4, 0]
        assert k[:2, 1].max() < k[2, 1] < k[3:, 1].min()

    def test_without_landmarks_falls_back_to_resized_crop(self, service, noise_image):
        aligned = service.align_face(noise_image, (100, 100, 300, 300))
        assert aligned.keypoints is None
        assert aligned.image.shape == (FACE_SIZE, FACE_SIZE, 3)


# --- extract_embedding_from_face --------------------------------------------------------------

class TestExtractEmbedding:
    def test_dimension_type_and_norm(self, service, valentin_photos):
        embedding = _embedding_of(service, valentin_photos[0])
        assert len(embedding) == EMBEDDING_DIM
        assert all(type(v) is float for v in embedding)  # floats de Python, no np.float32
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)  # normalizado L2

    def test_is_deterministic(self, service, valentin_photos):
        assert _embedding_of(service, valentin_photos[0]) == _embedding_of(service, valentin_photos[0])

    def test_reuses_precomputed_embedding(self, service):
        face = AlignedFace(bbox=[0, 0, 1, 1], keypoints=None, image=None, embedding=[0.5] * 4)
        assert service.extract_embedding_from_face(face) == [0.5] * 4

    def test_same_person_is_closer_than_different_people(self, service, valentin_photos, lucas_photos):
        val_a, val_b = (_embedding_of(service, p) for p in valentin_photos[:2])
        lucas = _embedding_of(service, lucas_photos[0])
        same, different = _cosine(val_a, val_b), _cosine(val_a, lucas)
        assert same > different
        assert same > SIMILARITY_THRESHOLD > different  # el umbral calibrado separa los dos casos


# --- identify ---------------------------------------------------------------------------------

class TestIdentify:
    def test_empty_store_is_unknown(self, service):
        assert service.identify([1.0] + [0.0] * (EMBEDDING_DIM - 1)) == ("unknown", 0.0)

    def test_best_match_above_threshold(self, service):
        query = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        other = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
        for label, embedding in (("ana", query), ("beto", other)):
            service.store.append(EmbeddingRecord(id_imagen=label, embedding=embedding, path="", etiqueta=label))
        label, score = service.identify(query)
        assert label == "ana" and score == pytest.approx(1.0)

    def test_below_threshold_is_unknown(self, service):
        service.store.append(
            EmbeddingRecord(id_imagen="x", embedding=[0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2), path="", etiqueta="ana")
        )
        label, score = service.identify([1.0] + [0.0] * (EMBEDDING_DIM - 1))
        assert label == "unknown" and score < SIMILARITY_THRESHOLD


# --- register_identity ------------------------------------------------------------------------

class TestRegisterIdentity:
    def test_registers_embedding_and_aligned_crop(self, service, valentin_photos, tmp_path):
        record = service.register_identity("Valentin", str(valentin_photos[0]), {})
        assert record.etiqueta == "Valentin"
        assert len(record.embedding) == EMBEDDING_DIM
        assert service.store.all() == [record]
        assert Path(record.path).parent == tmp_path and Path(record.path).exists()  # recorte alineado guardado

    def test_two_faces_fail_with_clear_message(self, service, two_faces_image, tmp_path):
        import cv2

        path = tmp_path / "dos.jpg"
        cv2.imwrite(str(path), two_faces_image)
        with pytest.raises(ValueError, match="Exactly one face"):
            service.register_identity("Alguien", str(path), {})
        assert service.store.all() == []

    def test_no_face_fails(self, service, noise_image, tmp_path):
        import cv2

        path = tmp_path / "ruido.jpg"
        cv2.imwrite(str(path), noise_image)
        with pytest.raises(ValueError, match="Exactly one face"):
            service.register_identity("Alguien", str(path), {})

    def test_unreadable_image_fails(self, service, tmp_path):
        with pytest.raises(ValueError, match="Could not read image"):
            service.register_identity("Alguien", str(tmp_path / "no_existe.jpg"), {})


# --- predict ----------------------------------------------------------------------------------

class TestPredict:
    @staticmethod
    def _predict(service, image_path, tmp_path) -> dict:
        result_file = service.predict(str(image_path), tmp_path / "resultados")
        return json.loads(open(result_file, encoding="utf-8").read())

    def test_registered_person_is_recognized(self, service, valentin_photos, tmp_path):
        for photo in valentin_photos[:3]:
            service.register_identity("Valentin", str(photo), {})
        result = self._predict(service, valentin_photos[3], tmp_path)
        (detection,) = result["detections"]
        assert detection["label"] == "Valentin"
        assert detection["score"] > SIMILARITY_THRESHOLD
        assert result["detected_people"] == ["Valentin"]
        assert sorted(detection["keypoints"]) == ["k0", "k1", "k2", "k3", "k4"]

    def test_unregistered_person_is_unknown(self, service, valentin_photos, lucas_photos, tmp_path):
        for photo in valentin_photos[:3]:
            service.register_identity("Valentin", str(photo), {})
        result = self._predict(service, lucas_photos[0], tmp_path)
        (detection,) = result["detections"]
        assert detection["label"] == "unknown"
        assert result["detected_people"] == []

    def test_image_without_faces_returns_empty_detections(self, service, noise_image, tmp_path):
        import cv2

        path = tmp_path / "ruido.jpg"
        cv2.imwrite(str(path), noise_image)
        result = self._predict(service, path, tmp_path)
        assert result["detections"] == [] and result["detected_people"] == []

    def test_several_people_in_one_image(self, service, valentin_photos, lucas_photos, two_faces_image, tmp_path):
        import cv2

        for photo in valentin_photos[:3]:
            service.register_identity("Valentin", str(photo), {})
        service.register_identity("Lucas", str(lucas_photos[0]), {})
        path = tmp_path / "grupo.jpg"
        cv2.imwrite(str(path), two_faces_image)
        result = self._predict(service, path, tmp_path)
        assert len(result["detections"]) == 2
        assert result["detected_people"] == ["Lucas", "Valentin"]
