"""Verificación (¿son la misma persona?): pares, ROC, EER, TAR@FAR y calibración del umbral.

Todo trabaja con embeddings normalizados L2: el coseno es un producto escalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class Pairs:
    """Pares de imágenes (posiciones i, j) y si son de la misma persona."""

    i: np.ndarray
    j: np.ndarray
    same: np.ndarray

    def __len__(self) -> int:
        return len(self.i)

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", embeddings[self.i], embeddings[self.j])


def make_pairs(labels: np.ndarray, n_per_class: int = 5000, seed: int = 42) -> Pairs:
    """n_per_class pares positivos y n_per_class negativos, sin repetir pares.

    Los positivos se reparten por igual entre identidades (si no, las que tienen más fotos
    aportarían más pares); los negativos son pares al azar de personas distintas.
    """
    rng = np.random.default_rng(seed)
    identities = np.unique(labels)
    per_identity = int(np.ceil(n_per_class / len(identities)))
    positives = []
    for identity in identities:
        positions = np.flatnonzero(labels == identity)
        a, b = np.triu_indices(len(positions), k=1)
        chosen = rng.choice(len(a), min(per_identity, len(a)), replace=False)
        positives.append(np.stack([positions[a[chosen]], positions[b[chosen]]], axis=1))
    positives = np.concatenate(positives)
    positives = positives[rng.permutation(len(positives))[:n_per_class]]

    negatives = set()
    while len(negatives) < n_per_class:
        i, j = rng.integers(0, len(labels), (2, n_per_class))
        for a, b in zip(i, j):
            if labels[a] != labels[b] and len(negatives) < n_per_class:
                negatives.add((min(a, b), max(a, b)))
    negatives = np.array(sorted(negatives))

    both = np.concatenate([positives, negatives])
    same = np.r_[np.ones(len(positives), bool), np.zeros(len(negatives), bool)]
    return Pairs(both[:, 0], both[:, 1], same)


def tar_at_far(far: np.ndarray, tar: np.ndarray, target_far: float) -> float:
    """Mayor TAR con FAR <= target_far (sobre la curva ROC)."""
    allowed = far <= target_far
    return float(tar[allowed].max()) if allowed.any() else 0.0


def threshold_at_far(scores: np.ndarray, same: np.ndarray, target_far: float) -> float:
    """Menor umbral cuyo FAR (fracción de impostores con score >= umbral) no supera target_far."""
    impostors = np.sort(scores[~same])[::-1]
    allowed = int(np.floor(target_far * len(impostors)))   # impostores que se pueden aceptar
    if allowed >= len(impostors):
        return float(impostors[-1])
    return float(np.nextafter(impostors[allowed], np.inf))


@dataclass(frozen=True)
class VerificationResult:
    auc: float
    eer: float
    eer_threshold: float
    tar_at_far: dict[float, float]
    far: np.ndarray = field(repr=False)
    tar: np.ndarray = field(repr=False)
    thresholds: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, float]:
        row = {"AUC": self.auc, "EER": self.eer, "umbral EER": self.eer_threshold}
        row.update({f"TAR@FAR={far:g}": tar for far, tar in self.tar_at_far.items()})
        return row


def verification_metrics(scores: np.ndarray, same: np.ndarray, fars: tuple[float, ...] = (1e-2, 1e-3)) -> VerificationResult:
    """ROC, AUC, EER (punto donde FAR = FRR) y TAR a los FAR pedidos."""
    far, tar, thresholds = roc_curve(same, scores)
    frr = 1 - tar
    k = int(np.argmin(np.abs(far - frr)))
    return VerificationResult(
        auc=float(roc_auc_score(same, scores)),
        eer=float((far[k] + frr[k]) / 2),
        eer_threshold=float(thresholds[k]),
        tar_at_far={target: tar_at_far(far, tar, target) for target in fars},
        far=far,
        tar=tar,
        thresholds=thresholds,
    )


def far_frr_curve(scores: np.ndarray, same: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """FAR, FRR, precision y recall de "misma persona" para cada umbral (score >= umbral acepta)."""
    accepted = scores[None, :] >= thresholds[:, None]
    true_accepts = (accepted & same).sum(axis=1)
    false_accepts = (accepted & ~same).sum(axis=1)
    return pd.DataFrame({
        "umbral": thresholds,
        "FAR": false_accepts / (~same).sum(),
        "FRR": 1 - true_accepts / same.sum(),
        "precision": np.divide(true_accepts, true_accepts + false_accepts,
                               out=np.ones(len(thresholds)), where=(true_accepts + false_accepts) > 0),
        "recall": true_accepts / same.sum(),
    })


def rate_at_threshold(scores: np.ndarray, same: np.ndarray, threshold: float) -> dict[str, float]:
    """FAR y FRR de un umbral fijo: sirve para ver cómo se comporta en test el umbral calibrado en validación."""
    accepted = scores >= threshold
    return {"umbral": threshold, "FAR": float(accepted[~same].mean()), "FRR": float(1 - accepted[same].mean())}


def hardest_errors(pairs: Pairs, scores: np.ndarray, threshold: float, top: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Falsos positivos con mayor score y falsos negativos con menor score al umbral dado."""
    frame = pd.DataFrame({"i": pairs.i, "j": pairs.j, "same": pairs.same, "score": scores})
    false_positives = frame[~frame["same"] & (frame["score"] >= threshold)].nlargest(top, "score")
    false_negatives = frame[frame["same"] & (frame["score"] < threshold)].nsmallest(top, "score")
    return false_positives.reset_index(drop=True), false_negatives.reset_index(drop=True)
