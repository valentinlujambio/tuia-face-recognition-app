"""Evaluación común de la Parte 2: todo lo que se calcula sobre los embeddings, sin importar el modelo."""

from lib.evaluation.embedding_store import SPLITS, EmbeddingSet, load_embeddings, results_path, save_embeddings

__all__ = ["SPLITS", "EmbeddingSet", "load_embeddings", "results_path", "save_embeddings"]
