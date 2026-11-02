from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRAIN, VAL_CLOSED, TEST_CLOSED, VAL_OPEN, TEST_OPEN = "train", "val_closed", "test_closed", "val_open", "test_open"
SPLIT_ORDER = [TRAIN, VAL_CLOSED, TEST_CLOSED, VAL_OPEN, TEST_OPEN]
OPEN_SPLITS = [VAL_OPEN, TEST_OPEN]


@dataclass(frozen=True)
class SplitConfig:
    """open_val/open_test: identidades enteras que el modelo nunca ve en entrenamiento.
    closed_val/closed_test: fracción de las imágenes de cada identidad conocida."""

    open_val_identities: int = 12
    open_test_identities: int = 13
    closed_val_fraction: float = 0.15
    closed_test_fraction: float = 0.15
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def near_duplicate_groups(
    embeddings: np.ndarray, labels: np.ndarray, threshold: float, block_size: int = 2048
) -> np.ndarray:
    """Un id de grupo por imagen: quedan en el mismo grupo las imágenes de una misma identidad unidas por
    una cadena de pares con coseno >= threshold (la misma foto con otro recorte, color o compresión).

    Los embeddings deben estar normalizados L2.
    """
    parent = np.arange(len(embeddings))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(embeddings), block_size):
        sims = embeddings[start : start + block_size] @ embeddings.T
        rows, cols = np.nonzero(sims >= threshold)
        rows = rows + start
        keep = (rows < cols) & (labels[rows] == labels[cols])
        for a, b in zip(rows[keep], cols[keep]):
            parent[find(a)] = find(b)
    return np.array([find(x) for x in range(len(embeddings))])


def _assign_groups(group_sizes: pd.Series, fractions: dict[str, float], rng: np.random.Generator) -> dict:
    """Reparte grupos enteros entre splits: cada grupo va al split más lejos de su cantidad objetivo."""
    total = group_sizes.sum()
    target = {split: fraction * total for split, fraction in fractions.items()}
    filled = dict.fromkeys(fractions, 0)
    assignment = {}
    for group in rng.permutation(group_sizes.index.to_numpy()):
        split = max(fractions, key=lambda s: (target[s] - filled[s]) / target[s])
        assignment[group] = split
        filled[split] += group_sizes[group]
    return assignment


def make_splits(df: pd.DataFrame, config: SplitConfig, groups: np.ndarray | None = None) -> pd.DataFrame:
    """Agrega las columnas split y scenario (closed/open). Conserva el índice de df.

    1) Separa identidades enteras para open-set (split por identidad).
    2) Con las restantes reparte las imágenes de cada identidad entre train, val_closed y test_closed
       (estratificado por identidad). Si se pasan groups, cada grupo de casi-duplicados va entero a un
       solo split, para que la misma foto no quede en train y en test.
    """
    rng = np.random.default_rng(config.seed)
    identities = np.array(sorted(df["identity"].unique()))
    rng.shuffle(identities)
    n_val, n_test = config.open_val_identities, config.open_test_identities
    if n_val + n_test >= len(identities):
        raise ValueError("No quedan identidades para entrenar.")
    open_val, open_test = set(identities[:n_val]), set(identities[n_val : n_val + n_test])

    out = df.copy()
    out["group"] = np.arange(len(df)) if groups is None else groups
    out["split"] = None
    out.loc[out["identity"].isin(open_val), "split"] = VAL_OPEN
    out.loc[out["identity"].isin(open_test), "split"] = TEST_OPEN

    fractions = {
        TRAIN: 1 - config.closed_val_fraction - config.closed_test_fraction,
        VAL_CLOSED: config.closed_val_fraction,
        TEST_CLOSED: config.closed_test_fraction,
    }
    closed = out[out["split"].isna()]
    for _, rows in closed.groupby("identity", sort=True):
        assignment = _assign_groups(rows["group"].value_counts().sort_index(), fractions, rng)
        out.loc[rows.index, "split"] = rows["group"].map(assignment)
    out["scenario"] = np.where(out["split"].isin(OPEN_SPLITS), "open", "closed")
    return out


def assert_no_group_leakage(df: pd.DataFrame) -> None:
    """Cada grupo de casi-duplicados está en un solo split."""
    splits_per_group = df.groupby("group")["split"].nunique()
    shared = int((splits_per_group > 1).sum())
    assert shared == 0, f"{shared} grupos de casi-duplicados quedaron repartidos en más de un split"


def identity_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Cantidad de identidades compartidas entre cada par de splits.

    Lo correcto: los splits closed comparten todas sus identidades entre sí, y val_open y test_open
    no comparten ninguna con nadie.
    """
    by_split = {split: set(df.loc[df["split"] == split, "identity"]) for split in SPLIT_ORDER}
    return pd.DataFrame(
        [[len(by_split[a] & by_split[b]) for b in SPLIT_ORDER] for a in SPLIT_ORDER],
        index=SPLIT_ORDER, columns=SPLIT_ORDER,
    )


def assert_no_identity_leakage(df: pd.DataFrame) -> None:
    """Open-set: ninguna identidad de val_open/test_open aparece en otro split."""
    overlaps = identity_overlaps(df)
    for split in OPEN_SPLITS:
        others = [s for s in SPLIT_ORDER if s != split]
        shared = int(overlaps.loc[split, others].sum())
        assert shared == 0, f"{split} comparte {shared} identidades con otros splits"


def assert_no_image_leakage(df: pd.DataFrame, column: str = "path") -> None:
    """Cada imagen está en un solo split."""
    repeated = df[column].duplicated(keep=False)
    assert not repeated.any(), f"{int(repeated.sum())} filas repiten {column} entre splits"


def nearest_in_reference(
    queries: np.ndarray, reference: np.ndarray, block_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Para cada query, la posición y el coseno de la imagen más parecida de reference.

    Los embeddings deben estar normalizados L2. Procesa por bloques para no armar la matriz completa.
    """
    positions, similarities = [], []
    for start in range(0, len(queries), block_size):
        sims = queries[start : start + block_size] @ reference.T
        best = sims.argmax(axis=1)
        positions.append(best)
        similarities.append(sims[np.arange(len(best)), best])
    return np.concatenate(positions), np.concatenate(similarities)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Identidades, imágenes y porcentaje de imágenes por split, con fila de total."""
    summary = df.groupby("split").agg(escenario=("scenario", "first"), identidades=("identity", "nunique"),
                                      imagenes=("path", "size"))
    summary = summary.reindex(SPLIT_ORDER)
    summary["% imagenes"] = (100 * summary["imagenes"] / len(df)).round(1)
    summary.loc["total"] = ["", df["identity"].nunique(), len(df), 100.0]
    return summary
