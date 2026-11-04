from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lib.vision import Detection, FaceAligner, FaceDetector

KEYPOINT_COLUMNS = [f"kp{i}_{axis}" for i in range(5) for axis in ("x", "y")]


def _face_row(detections: list[Detection]) -> dict:
    if not detections:
        return {"n_faces": 0}
    main = detections[0]
    second_area = detections[1].area if len(detections) > 1 else 0
    return {
        "n_faces": len(detections),
        "det_score": main.score,
        "x1": main.bbox[0], "y1": main.bbox[1], "x2": main.bbox[2], "y2": main.bbox[3],
        "face_min_side": main.min_side,
        "face_area_ratio": main.area / second_area if second_area else np.inf,
        **dict(zip(KEYPOINT_COLUMNS, main.keypoints.reshape(-1).tolist())),
    }


def detect_and_align(
    df: pd.DataFrame,
    root: Path,
    detector: FaceDetector,
    aligner: FaceAligner,
    aligned_root: Path,
) -> pd.DataFrame:
    """Detecta, alinea la cara principal (la de mayor área) y guarda el recorte en aligned_root/<path>.jpg.

    Agrega por imagen: cantidad de caras, score, caja y landmarks de la principal,
    lado menor de la caja, relación de área con la segunda cara y ruta del recorte.
    Los archivos que no abren quedan con n_faces = NaN.
    """
    rows = []
    for rel_path, readable in tqdm(zip(df["path"], df["readable"]), total=len(df), desc="Detectando"):
        row: dict = {"n_faces": np.nan}
        image = cv2.imread(str(Path(root) / rel_path)) if readable else None
        if image is not None:
            detections = detector.detect(image)
            row = _face_row(detections)
            if detections:
                aligned_path = Path(rel_path).with_suffix(".jpg").as_posix()
                target = Path(aligned_root) / aligned_path
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), aligner.align(image, detections[0].keypoints), [cv2.IMWRITE_JPEG_QUALITY, 95])
                row["aligned_path"] = aligned_path
        rows.append(row)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def keypoints_of(row: pd.Series) -> np.ndarray:
    return row[KEYPOINT_COLUMNS].to_numpy(dtype=np.float32).reshape(5, 2)


def pose_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Pose aproximada a partir de los 5 landmarks.

    roll: inclinación de la línea de los ojos, en grados.
    yaw_proxy: desplazamiento horizontal de la nariz respecto del punto medio de los ojos,
    dividido por la distancia entre ojos. 0 es frontal; el signo indica hacia qué lado gira.
    """
    dx, dy = df["kp1_x"] - df["kp0_x"], df["kp1_y"] - df["kp0_y"]
    eye_distance = np.hypot(dx, dy).clip(lower=1e-6)
    eyes_mid_x = (df["kp0_x"] + df["kp1_x"]) / 2
    return pd.DataFrame(
        {"roll": np.degrees(np.arctan2(dy, dx)), "yaw_proxy": (df["kp2_x"] - eyes_mid_x) / eye_distance},
        index=df.index,
    )


def load_aligned(paths: pd.Series | list[str], aligned_root: Path) -> list[np.ndarray]:
    return [cv2.imread(str(Path(aligned_root) / p)) for p in paths]
