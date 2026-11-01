from __future__ import annotations

import numpy as np
import pandas as pd


def similar_pairs(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray | None = None,
    threshold: float = 0.95,
    block_size: int = 2048,
) -> pd.DataFrame:
    """Pares (i, j, similitud) con coseno >= threshold. Los embeddings deben estar normalizados L2.

    Sin embeddings_b compara a contra sí mismo y devuelve solo i < j.
    Procesa por bloques para no armar la matriz N×N completa en memoria.
    """
    same = embeddings_b is None
    embeddings_b = embeddings_a if same else embeddings_b
    rows = []
    for start in range(0, len(embeddings_a), block_size):
        sims = embeddings_a[start : start + block_size] @ embeddings_b.T
        i_idx, j_idx = np.nonzero(sims >= threshold)
        i_idx = i_idx + start
        if same:
            keep = i_idx < j_idx
            i_idx, j_idx = i_idx[keep], j_idx[keep]
        rows.append(pd.DataFrame({"i": i_idx, "j": j_idx, "similarity": sims[i_idx - start, j_idx]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["i", "j", "similarity"])


def duplicates_to_drop(pairs: pd.DataFrame, n_items: int) -> np.ndarray:
    """Máscara de elementos a descartar: de cada grupo de duplicados se conserva el de menor índice.

    Los grupos se arman con union-find: si a ~ b y b ~ c, los tres son un grupo aunque a y c
    no superen el umbral entre sí.
    """
    parent = np.arange(n_items)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(pairs["i"], pairs["j"]):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(k) for k in range(n_items)])
    return roots != np.arange(n_items)
