from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def project_tsne(embeddings: np.ndarray, pca_dims: int = 50, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """Proyección 2D: PCA a pca_dims (saca ruido y acelera) y después t-SNE.

    Con embeddings normalizados L2, la distancia euclídea que usa t-SNE es monótona con el coseno.
    """
    reduced = PCA(n_components=min(pca_dims, *embeddings.shape), random_state=seed).fit_transform(embeddings)
    perplexity = min(perplexity, (len(embeddings) - 1) / 3)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(reduced)
