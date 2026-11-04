from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
import onnxruntime
from lib.schemas import EmbeddingRecord, FaceDetection, PredictResult, AlignedFace
from lib.storage.base import EmbeddingStoreProtocol
from lib.vision import FaceAligner, InputNormalization, OnnxFaceEmbedder, ScrfdDetector, l2_normalize
import os
import logging

logger = logging.getLogger(__name__)

# IoU mínimo entre la caja pedida y la que se vuelve a detectar en el recorte con margen
# para tomar sus landmarks como los de esa cara.
MIN_LANDMARK_IOU = 0.3


class FaceService:
    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        face_size: int,
        model_path: Path,
        output_path: Path = Path("output"),
        detector_path: Path | None = None,
        det_score_threshold: float = 0.5,
        det_input_size: int = 640,
        min_face_size: int = 20,
        align_margin: float = 0.5,
        det_pad_ratio: float = 0.25,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.face_size = face_size
        # Mismo preprocesamiento que en el entrenamiento y la evaluación (lib.vision)
        self.normalization = InputNormalization()
        self.aligner = FaceAligner(face_size)
        self.model: any = self._load_model(model_path)
        self.output_path = output_path
        self.det_score_threshold = det_score_threshold
        self.det_input_size = det_input_size
        self.min_face_size = min_face_size
        self.align_margin = align_margin
        self.det_pad_ratio = det_pad_ratio
        self.detector: any = self._load_detector(detector_path)

        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    @staticmethod
    def _kps_to_keypoints_dict(kps: np.ndarray | None) -> dict[str, list[int]]:
        if kps is None or len(kps) == 0:
            return {}
        return {
            f"k{i}": [int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))]
            for i in range(len(kps))
        }


    def _load_model(self, model_path: Path) -> any:
        mp = Path(model_path)
        if not mp.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        suf = mp.suffix.lower()
        if suf == ".pth":
            return torch.load(mp, map_location="cpu", weights_only=False)
        if suf == ".onnx":
            # Reconocedor ONNX (ArcFace w600k_r50 o la CNN custom exportada): sesión de onnxruntime
            # con la normalización de entrada del entrenamiento y salida normalizada L2.
            return OnnxFaceEmbedder(mp, normalization=self.normalization)
        raise ValueError(f"Unsupported model format (expected .pth or .onnx): {model_path}")

    def _load_detector(self, detector_path: Path | None) -> ScrfdDetector:
        """
        Detector SCRFD de InsightFace (det_10g.onnx de buffalo_l), el mismo de la Parte 1.
        Se carga una sola vez. Con DETECTOR_MODEL_NAME se usa ese archivo de MODEL_PATH;
        sin él, el de ~/.insightface (se descarga buffalo_l si no está).
        """
        kwargs = dict(
            score_threshold=self.det_score_threshold,
            input_size=self.det_input_size,
            min_face_size=self.min_face_size,
            pad_ratio=self.det_pad_ratio,
        )
        if detector_path is None:
            return ScrfdDetector.from_model_pack(**kwargs)
        if not Path(detector_path).exists():
            raise ValueError(f"Detector path does not exist: {detector_path}")
        return ScrfdDetector(detector_path, **kwargs)

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(source_path)
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (InsightFace / OpenCV convention)
        return image
    
    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """Intersection over union between two boxes (x1, y1, x2, y2)."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0
    
    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Each box is (x1, y1, x2, y2) in pixels (InsightFace convention).
        Return a list of tuples with the coordinates of the faces detected in the image.

        El detector ya filtra por DET_SCORE_THRESHOLD y MIN_FACE_SIZE y ordena de mayor a menor área.
        Sin caras devuelve una lista vacía.
        """
        height, width = image.shape[:2]
        return [self._clip_xyxy(*detection.bbox, height, width) for detection in self.detector.detect(image)]

    def align_face(
        self, image: np.ndarray, box: tuple[int, int, int, int]
    ) -> AlignedFace:
        """
        Crop using box (x1, y1, x2, y2) and run FaceAnalysis on the crop.
        Return an AlignedFace object.

        Los landmarks se vuelven a detectar en un recorte con ALIGN_MARGIN de margen alrededor de la caja
        (no en el recorte justo, donde SCRFD pierde contexto) y se toma la detección que más se superpone
        con la caja. Con ellos se alinea la imagen completa a la plantilla de ArcFace, como en la Parte 1.
        Los keypoints se devuelven relativos a la esquina superior izquierda de la caja.
        """
        x1, y1, x2, y2 = (int(v) for v in box)
        height, width = image.shape[:2]
        margin = int(round(self.align_margin * max(x2 - x1, y2 - y1)))
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(width, x2 + margin), min(height, y2 + margin)
        offset = np.array([cx1, cy1], dtype=np.float32)

        target = np.array([x1, y1, x2, y2], dtype=np.float32)
        best, best_iou = None, 0.0
        for detection in self.detector.detect(image[cy1:cy2, cx1:cx2]):
            iou = self._iou(np.asarray(detection.bbox, dtype=np.float32) + np.tile(offset, 2), target)
            if iou > best_iou:
                best, best_iou = detection, iou

        if best is None or best_iou < MIN_LANDMARK_IOU:
            # Sin landmarks no se puede alinear: recorte de la caja redimensionado (el frontend dibuja solo la caja)
            logger.warning(f"No landmarks found for box {box}; using the resized crop")
            crop = cv2.resize(image[y1:y2, x1:x2], (self.face_size, self.face_size))
            return AlignedFace(bbox=[x1, y1, x2, y2], keypoints=None, image=crop)

        keypoints = best.keypoints + offset
        return AlignedFace(
            bbox=[x1, y1, x2, y2],
            keypoints=keypoints - np.array([x1, y1], dtype=np.float32),
            image=self.aligner.align(image, keypoints),
        )

    def extract_embedding_from_face(self, face: AlignedFace) -> list[float]:
        """
        Extract embedding from face.
        Return a list of floats representing the embedding of the face.

        Mismo preprocesamiento que en el entrenamiento (InputNormalization: BGR -> RGB, (x - 127.5) / 127.5,
        HWC -> NCHW) y normalización L2, con .onnx (OnnxFaceEmbedder) o con un modelo .pth completo.
        """
        if face.embedding is not None:
            return face.embedding
        if isinstance(self.model, OnnxFaceEmbedder):
            embedding = self.model.embed([face.image])[0]
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.from_numpy(self.normalization([face.image])))
            embedding = l2_normalize(output.cpu().numpy())[0]
        return embedding.astype(float).tolist()
        
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    def _candidates(self, query_embedding: list[float]) -> list[EmbeddingRecord]:
        """
        Con pgvector y coseno, la base busca el registro más parecido (operador <=>).
        Con el store JSON o con l2 se recorren todos, porque <=> es distancia coseno.
        """
        if hasattr(self.store, "search") and self.similarity_metric.lower() != "l2":
            return self.store.search(query_embedding, k=1)
        return self.store.all()

    def identify(self, query_embedding: list[float]) -> tuple[str, float]:
        records = self._candidates(query_embedding)
        if not records:
            return "unknown", 0.0

        best_label = "unknown"
        best_score = -1.0
        for record in records:
            score = self.similarity(query_embedding, record.embedding)
            if score > best_score:
                best_score = score
                best_label = record.etiqueta

        if best_score < self.similarity_threshold:
            return "unknown", max(best_score, 0.0)
        return best_label, best_score

    def register_identity(
        self, identity: str, image_path: str, metadata: dict[str, object]
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        faces = self.detect_faces(image)

        if len(faces) != 1:
            raise ValueError("Exactly one face must be detected for identity registration.")
        
        logger.info(f"Face detected: {faces[0]}")

        box = faces[0]
        aligned = self.align_face(image, box)
        embedding = self.extract_embedding_from_face(aligned)

        img_id = str(uuid4())
        img_output_path = self.output_path / f"img_{img_id}.jpg"
        
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(img_output_path),
            etiqueta=identity,
            metadata=metadata,
        )
        self.store.append(record)

        cv2.imwrite(str(img_output_path), aligned.image)
        logger.info(f"Identity registered: {identity} with image: {image_path}")
        return record

    def predict(self, source_path: str, output_path: Path) -> str:
        image = self._load_image(source_path)
        faces = self.detect_faces(image)
        detections: list[FaceDetection] = []
        for (x1, y1, x2, y2) in faces:
            aligned = self.align_face(image, (x1, y1, x2, y2))
            embedding = self.extract_embedding_from_face(aligned)
            label, score = self.identify(embedding)
            kps = getattr(aligned, "keypoints", None)
            kps_arr = np.asarray(kps) if kps is not None else None
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    keypoints=self._kps_to_keypoints_dict(kps_arr),
                    label=label,
                    score=round(float(score), 4),
                )
            )

        detected_people = sorted({item.label for item in detections if item.label != "unknown"})
        result_payload = PredictResult(
            source_path=source_path,
            detections=detections,
            detected_people=detected_people,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(result_payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
