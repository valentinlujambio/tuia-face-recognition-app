"""Identificación (¿quién es?): galería, probes, Top-k y rechazo de desconocidos.

Imita a la app: la galería son unas pocas imágenes registradas por identidad (como `/insert`)
y cada probe toma la identidad del registro más parecido (`FaceService.identify`). Si ese score
queda por debajo del umbral, la respuesta es `unknown`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

UNKNOWN = "unknown"


def split_gallery(labels: np.ndarray, per_identity: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de galería (per_identity por identidad, al azar) y de probes (el resto)."""
    rng = np.random.default_rng(seed)
    gallery = np.concatenate([
        rng.choice(np.flatnonzero(labels == identity), per_identity, replace=False)
        for identity in np.unique(labels)
    ])
    probes = np.setdiff1d(np.arange(len(labels)), gallery)
    return np.sort(gallery), probes


def identity_scores(
    gallery_emb: np.ndarray, gallery_labels: np.ndarray, probe_emb: np.ndarray, aggregate: str = "max"
) -> tuple[np.ndarray, np.ndarray]:
    """(identidades, scores (n_probes, n_identidades)).

    aggregate="max": score del registro más parecido de cada identidad, como `identify` de la app.
    aggregate="mean": coseno con el promedio de los embeddings de la identidad (plantilla).
    """
    identities = np.unique(gallery_labels)
    if aggregate == "max":
        sims = probe_emb @ gallery_emb.T
        scores = np.stack([sims[:, gallery_labels == identity].max(axis=1) for identity in identities], axis=1)
    elif aggregate == "mean":
        templates = np.stack([gallery_emb[gallery_labels == identity].mean(axis=0) for identity in identities])
        templates /= np.linalg.norm(templates, axis=1, keepdims=True)
        scores = probe_emb @ templates.T
    else:
        raise ValueError(f"aggregate desconocido: {aggregate}")
    return identities, scores


def top_k_accuracy(identities: np.ndarray, scores: np.ndarray, true_labels: np.ndarray, k: int) -> float:
    """Fracción de probes cuya identidad está entre las k de mayor score (sin umbral)."""
    top = identities[np.argsort(-scores, axis=1)[:, :k]]
    return float((top == true_labels[:, None]).any(axis=1).mean())


def predict(identities: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(predicción, mejor score). Por debajo del umbral, `unknown`."""
    best = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(scores)), best]
    return np.where(best_scores >= threshold, identities[best], UNKNOWN), best_scores


@dataclass(frozen=True)
class IdentificationResult:
    top1: float
    top5: float
    precision: float
    recall: float
    f1: float
    known_accepted_correct: float
    known_rejected: float
    known_misidentified: float
    unknown_rejected: float
    predictions: pd.DataFrame

    def summary(self) -> dict[str, float]:
        return {
            "Top-1": self.top1, "Top-5": self.top5,
            "precision (macro)": self.precision, "recall (macro)": self.recall, "F1 (macro)": self.f1,
            "conocidos: correctos": self.known_accepted_correct,
            "conocidos: rechazados": self.known_rejected,
            "conocidos: otra identidad": self.known_misidentified,
            "desconocidos rechazados": self.unknown_rejected,
        }


def evaluate_identification(
    gallery_emb: np.ndarray,
    gallery_labels: np.ndarray,
    probe_emb: np.ndarray,
    probe_labels: np.ndarray,
    threshold: float,
    unknown_emb: np.ndarray | None = None,
    unknown_labels: np.ndarray | None = None,
    aggregate: str = "max",
) -> IdentificationResult:
    """Identificación con galería, probes conocidos y probes desconocidos.

    - Top-1 y Top-5: solo probes conocidos y sin umbral (closed-set puro).
    - Precision, recall y F1 macro: con umbral, sobre conocidos y desconocidos juntos. La clase
      verdadera de un desconocido es `unknown`; el promedio macro es sobre las identidades
      registradas, así aceptar a un desconocido como alguien baja la precision de esa persona.
    - Desconocidos rechazados: fracción de probes no registrados que la app devuelve como `unknown`.
    """
    identities, scores = identity_scores(gallery_emb, gallery_labels, probe_emb, aggregate)
    top1 = top_k_accuracy(identities, scores, probe_labels, 1)
    top5 = top_k_accuracy(identities, scores, probe_labels, 5)
    known_pred, known_best = predict(identities, scores, threshold)

    true_labels, predicted, best_scores = [probe_labels], [known_pred], [known_best]
    unknown_rejected = np.nan
    if unknown_emb is not None and len(unknown_emb):
        _, unknown_scores = identity_scores(gallery_emb, gallery_labels, unknown_emb, aggregate)
        unknown_pred, unknown_best = predict(identities, unknown_scores, threshold)
        unknown_rejected = float((unknown_pred == UNKNOWN).mean())
        true_labels.append(np.full(len(unknown_emb), UNKNOWN))
        predicted.append(unknown_pred)
        best_scores.append(unknown_best)
        source = np.r_[np.full(len(probe_emb), "conocido"), np.full(len(unknown_emb), "desconocido")]
        original = np.r_[probe_labels, unknown_labels if unknown_labels is not None else np.full(len(unknown_emb), UNKNOWN)]
    else:
        source = np.full(len(probe_emb), "conocido")
        original = probe_labels

    true_labels, predicted = np.concatenate(true_labels), np.concatenate(predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted, labels=identities, average="macro", zero_division=0
    )
    return IdentificationResult(
        top1=top1,
        top5=top5,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        known_accepted_correct=float((known_pred == probe_labels).mean()),
        known_rejected=float((known_pred == UNKNOWN).mean()),
        known_misidentified=float(((known_pred != probe_labels) & (known_pred != UNKNOWN)).mean()),
        unknown_rejected=unknown_rejected,
        predictions=pd.DataFrame({
            "tipo": source, "identidad": original, "verdadera": true_labels,
            "predicha": predicted, "score": np.concatenate(best_scores),
        }),
    )


def best_scores(
    gallery_emb: np.ndarray, gallery_labels: np.ndarray, probe_emb: np.ndarray, probe_labels: np.ndarray | None = None,
    aggregate: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """(mejor score de cada probe, si la mejor identidad es la correcta). Sin labels, todo False (desconocidos)."""
    identities, scores = identity_scores(gallery_emb, gallery_labels, probe_emb, aggregate)
    best = scores.argmax(axis=1)
    correct = identities[best] == probe_labels if probe_labels is not None else np.zeros(len(best), bool)
    return scores[np.arange(len(scores)), best], correct


def open_set_curve(known_best: np.ndarray, known_correct: np.ndarray, unknown_best: np.ndarray,
                   thresholds: np.ndarray) -> pd.DataFrame:
    """Por umbral: conocidos identificados bien (DIR) y desconocidos aceptados como alguien (FPIR)."""
    return pd.DataFrame({
        "umbral": thresholds,
        "DIR": ((known_best[None, :] >= thresholds[:, None]) & known_correct).mean(axis=1),
        "FPIR": (unknown_best[None, :] >= thresholds[:, None]).mean(axis=1),
    })


def threshold_at_fpir(unknown_best: np.ndarray, target_fpir: float) -> float:
    """Menor umbral que acepta como alguien registrado a lo sumo target_fpir de los desconocidos."""
    ordered = np.sort(unknown_best)[::-1]
    allowed = int(np.floor(target_fpir * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def most_confused(predictions: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Pares (verdadera → predicha) más frecuentes entre los errores, sin contar rechazos."""
    errors = predictions[(predictions["verdadera"] != predictions["predicha"]) & (predictions["predicha"] != UNKNOWN)]
    return (errors.groupby(["identidad", "predicha"]).size().rename("veces")
            .sort_values(ascending=False).head(top).reset_index())
