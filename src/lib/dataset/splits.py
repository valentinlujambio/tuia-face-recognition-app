from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def make_splits(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Agrega las columnas split y scenario (closed/open). Conserva el índice de df.

    1) Separa identidades enteras para open-set (split por identidad).
    2) Con las restantes divide las imágenes estratificando por identidad (closed-set).
    """
    rng = np.random.default_rng(config.seed)
    identities = np.array(sorted(df["identity"].unique()))
    rng.shuffle(identities)
    n_val, n_test = config.open_val_identities, config.open_test_identities
    if n_val + n_test >= len(identities):
        raise ValueError("No quedan identidades para entrenar.")
    open_val, open_test = set(identities[:n_val]), set(identities[n_val : n_val + n_test])

    out = df.copy()
    out["split"] = None
    out.loc[out["identity"].isin(open_val), "split"] = VAL_OPEN
    out.loc[out["identity"].isin(open_test), "split"] = TEST_OPEN

    closed = out[out["split"].isna()]
    holdout_fraction = config.closed_val_fraction + config.closed_test_fraction
    train_idx, holdout_idx = train_test_split(
        closed.index, test_size=holdout_fraction, stratify=closed["identity"], random_state=config.seed
    )
    holdout = closed.loc[holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout.index,
        test_size=config.closed_test_fraction / holdout_fraction,
        stratify=holdout["identity"],
        random_state=config.seed,
    )
    out.loc[train_idx, "split"] = TRAIN
    out.loc[val_idx, "split"] = VAL_CLOSED
    out.loc[test_idx, "split"] = TEST_CLOSED
    out["scenario"] = np.where(out["split"].isin(OPEN_SPLITS), "open", "closed")
    return out


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
