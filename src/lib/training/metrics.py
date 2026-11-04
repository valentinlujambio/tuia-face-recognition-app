"""Métricas que se calculan durante el entrenamiento (la evaluación completa está en lib.evaluation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class VerificationScore:
    auc: float
    eer: float
    eer_threshold: float


def verification_score(embeddings: np.ndarray, labels: np.ndarray) -> VerificationScore:
    """AUC y EER de verificación con todos los pares posibles (coseno entre embeddings L2).

    Con ~2000 caras son ~2M pares: entra en memoria y no depende de cómo se muestrean los pares.
    """
    labels = np.asarray(labels)
    rows, cols = np.triu_indices(len(labels), k=1)
    similarities = np.einsum("ij,ij->i", embeddings[rows], embeddings[cols])
    same = labels[rows] == labels[cols]
    far, tar, thresholds = roc_curve(same, similarities)
    frr = 1.0 - tar
    i = int(np.nanargmin(np.abs(far - frr)))
    return VerificationScore(
        auc=float(roc_auc_score(same, similarities)),
        eer=float((far[i] + frr[i]) / 2),
        eer_threshold=float(thresholds[i]),
    )
