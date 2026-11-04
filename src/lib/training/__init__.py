"""Entrenamiento del Modelo A (datos, loop, checkpoints y métricas de seguimiento)."""

from lib.training.data import FaceDataset, LabelEncoder, load_faces
from lib.training.experiment import Experiment, build_head, build_optimizer, build_scheduler, run_experiment
from lib.training.metrics import VerificationScore, verification_score
from lib.training.trainer import CheckpointStore, EarlyStopping, EpochLog, Trainer, set_seed

__all__ = [
    "CheckpointStore",
    "EarlyStopping",
    "EpochLog",
    "Experiment",
    "FaceDataset",
    "LabelEncoder",
    "Trainer",
    "VerificationScore",
    "build_head",
    "build_optimizer",
    "build_scheduler",
    "load_faces",
    "run_experiment",
    "set_seed",
    "verification_score",
]
