from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ClusterFactory = Callable[[int], object]


def project_tsne(embeddings: np.ndarray, pca_dims: int = 50, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """Proyección 2D: PCA a pca_dims (saca ruido y acelera) y después t-SNE.

    Con embeddings normalizados L2, la distancia euclídea que usa t-SNE es monótona con el coseno.
    """
    reduced = PCA(n_components=min(pca_dims, *embeddings.shape), random_state=seed).fit_transform(embeddings)
    perplexity = min(perplexity, (len(embeddings) - 1) / 3)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(reduced)


def sample_per_identity(labels: np.ndarray, per_identity: int, seed: int = 42) -> np.ndarray:
    """Posiciones de una muestra con la misma cantidad de imágenes por identidad (o todas, si tiene menos)."""
    rng = np.random.default_rng(seed)
    picks = []
    for identity in np.unique(labels):
        positions = np.flatnonzero(labels == identity)
        picks.append(rng.choice(positions, min(per_identity, len(positions)), replace=False))
    return np.sort(np.concatenate(picks))


def same_vs_different_similarities(
    embeddings: np.ndarray, labels: np.ndarray, n_different: int = 200_000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Cosenos entre fotos de la misma persona (todos los pares) y de personas distintas (pares al azar).

    Los embeddings deben estar normalizados L2.
    """
    same = []
    for identity in np.unique(labels):
        group = embeddings[labels == identity]
        same.append((group @ group.T)[np.triu_indices(len(group), k=1)])
    i, j = np.random.default_rng(seed).integers(0, len(embeddings), (2, n_different))
    different = i[labels[i] != labels[j]], j[labels[i] != labels[j]]
    return np.concatenate(same), np.einsum("ij,ij->i", embeddings[different[0]], embeddings[different[1]])


def default_algorithms(seed: int = 42, dbscan_eps: float = 0.5, dbscan_min_samples: int = 5) -> dict[str, ClusterFactory]:
    """Algoritmos a comparar. Cada fábrica recibe k (cantidad de identidades); DBSCAN no lo usa."""
    return {
        "K-Means": lambda k: KMeans(n_clusters=k, n_init=10, random_state=seed),
        "Aglomerativo (coseno, average)": lambda k: AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average"),
        f"DBSCAN (coseno, eps={dbscan_eps:g})": lambda k: DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine"),
    }


def evaluate_clustering(
    embeddings: np.ndarray, labels: np.ndarray, algorithms: Mapping[str, ClusterFactory]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """ARI y NMI contra las etiquetas reales y silhouette (coseno) de cada algoritmo.

    La primera fila es el silhouette de las etiquetas reales: cuán separables son
    las identidades en este espacio, la referencia para comparar los algoritmos.
    El ruido de DBSCAN (-1) cuenta como un cluster más para ARI y NMI y se excluye del silhouette.
    """
    k = len(np.unique(labels))
    rows = [{"algoritmo": "Etiquetas reales", "clusters": k, "ruido": 0, "ARI": 1.0, "NMI": 1.0,
             "silhouette": silhouette_score(embeddings, labels, metric="cosine")}]
    assignments = {}
    for name, factory in algorithms.items():
        predicted = factory(k).fit_predict(embeddings)
        assignments[name] = predicted
        clustered = predicted != -1
        n_clusters = len(np.unique(predicted[clustered]))
        rows.append({
            "algoritmo": name,
            "clusters": n_clusters,
            "ruido": int((~clustered).sum()),
            "ARI": adjusted_rand_score(labels, predicted),
            "NMI": normalized_mutual_info_score(labels, predicted),
            "silhouette": (
                silhouette_score(embeddings[clustered], predicted[clustered], metric="cosine")
                if 1 < n_clusters < clustered.sum() else np.nan
            ),
        })
    return pd.DataFrame(rows), assignments


def identity_centroids(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(identidades, centroides normalizados L2)."""
    identities = np.unique(labels)
    centroids = np.stack([embeddings[labels == identity].mean(axis=0) for identity in identities])
    return identities, centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def label_audit(embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Por imagen: coseno con el centroide de su identidad y con el de la identidad ajena más parecida.

    Similitud propia baja: la imagen está lejos de su clase (outlier).
    Similitud alta con otra identidad: probablemente es otra persona del dataset (error de etiqueta).
    """
    identities, centroids = identity_centroids(embeddings, labels)
    sims = embeddings @ centroids.T
    rows = np.arange(len(embeddings))
    own = np.searchsorted(identities, labels)
    own_sim = sims[rows, own]
    sims[rows, own] = -np.inf
    closest = sims.argmax(axis=1)
    return pd.DataFrame({
        "identity": labels,
        "own_similarity": own_sim,
        "closest_other": identities[closest],
        "closest_other_similarity": sims[rows, closest],
    })


def condition_neighbor_rate(
    embeddings: np.ndarray, labels: np.ndarray, flag: np.ndarray, min_images: int = 3
) -> tuple[float, float, int]:
    """Si las imágenes con una condición de captura (flag) forman un subgrupo dentro de su identidad.

    Para cada imagen con la condición busca su vecina más parecida de la misma identidad y mide
    con qué frecuencia también tiene la condición. Se compara con lo esperable al azar, que es
    la proporción de la condición en esa identidad. Solo cuenta identidades con al menos
    min_images imágenes con y sin la condición.
    Devuelve (frecuencia observada, frecuencia al azar, identidades evaluadas).
    """
    observed, chance = [], []
    for identity in np.unique(labels):
        mask = labels == identity
        has = flag[mask]
        if has.sum() < min_images or (~has).sum() < min_images:
            continue
        group = embeddings[mask]
        sims = group @ group.T
        np.fill_diagonal(sims, -np.inf)
        nearest = sims.argmax(axis=1)
        observed.append(has[nearest][has].mean())
        chance.append((has.sum() - 1) / (len(has) - 1))
    return float(np.mean(observed)), float(np.mean(chance)), len(observed)


def closest_identity_pairs(embeddings: np.ndarray, labels: np.ndarray, top: int = 10) -> pd.DataFrame:
    """Pares de identidades con centroides más parecidos: las que más se pueden mezclar."""
    identities, centroids = identity_centroids(embeddings, labels)
    sims = centroids @ centroids.T
    i_idx, j_idx = np.triu_indices(len(identities), k=1)
    pairs = pd.DataFrame({"identidad_a": identities[i_idx], "identidad_b": identities[j_idx], "similitud_centroides": sims[i_idx, j_idx]})
    return pairs.nlargest(top, "similitud_centroides").reset_index(drop=True)


def split_identities(labels: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Identidades repartidas en más de un cluster (sin contar el ruido) y qué fracción cae en su cluster mayoritario."""
    frame = pd.DataFrame({"identity": labels, "cluster": predicted})
    grouped = frame[frame["cluster"] != -1].groupby("identity")["cluster"]
    purity = pd.DataFrame({
        "clusters": grouped.nunique(),
        "fraccion_mayoritario": grouped.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
    })
    return purity[purity["clusters"] > 1].sort_values("fraccion_mayoritario")


def merged_clusters(labels: np.ndarray, predicted: np.ndarray, min_images: int = 5) -> pd.DataFrame:
    """Clusters que juntan al menos min_images imágenes de dos o más identidades."""
    frame = pd.DataFrame({"identity": labels, "cluster": predicted})
    rows = []
    for cluster, members in frame[frame["cluster"] != -1].groupby("cluster")["identity"]:
        counts = members.value_counts()
        counts = counts[counts >= min_images]
        if len(counts) > 1:
            rows.append({"cluster": cluster, "identidades": ", ".join(f"{name} ({n})" for name, n in counts.items())})
    return pd.DataFrame(rows, columns=["cluster", "identidades"])
