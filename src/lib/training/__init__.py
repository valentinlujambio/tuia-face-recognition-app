"""Entrenamiento del Modelo A (datos, loop, checkpoints y métricas de seguimiento)."""

from lib.training.data import FaceDataset, LabelEncoder, load_faces
from lib.training.metrics import VerificationScore, verification_score
from lib.training.trainer import CheckpointStore, EarlyStopping, EpochLog, Trainer, set_seed

__all__ = [
    "CheckpointStore",
    "EarlyStopping",
    "EpochLog",
    "FaceDataset",
    "LabelEncoder",
    "Trainer",
    "VerificationScore",
    "load_faces",
    "set_seed",
    "verification_score",
]
