"""Evaluación común de la Parte 2: todo lo que se calcula sobre los embeddings, sin importar el modelo."""

from lib.evaluation.embedding_store import SPLITS, EmbeddingSet, load_embeddings, results_path, save_embeddings
from lib.evaluation.identification import (UNKNOWN, IdentificationResult, best_scores, evaluate_identification,
                                           most_confused, open_set_curve, split_gallery, threshold_at_fpir)
from lib.evaluation.verification import (Pairs, VerificationResult, far_frr_curve, hardest_errors, make_pairs,
                                         rate_at_threshold, threshold_at_far, verification_metrics)

__all__ = [
    "SPLITS",
    "UNKNOWN",
    "EmbeddingSet",
    "IdentificationResult",
    "Pairs",
    "VerificationResult",
    "best_scores",
    "evaluate_identification",
    "far_frr_curve",
    "hardest_errors",
    "load_embeddings",
    "make_pairs",
    "most_confused",
    "open_set_curve",
    "rate_at_threshold",
    "results_path",
    "save_embeddings",
    "split_gallery",
    "threshold_at_far",
    "threshold_at_fpir",
    "verification_metrics",
]
