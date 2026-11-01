from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class QualityRule(ABC):
    """Regla de descarte reproducible sobre las columnas del DataFrame del dataset."""

    name: str

    @abstractmethod
    def fails(self, df: pd.DataFrame) -> pd.Series:
        """Serie booleana: True si la fila debe descartarse."""

    def describe(self) -> str:
        return self.name


@dataclass
class CorruptFileRule(QualityRule):
    name: str = "archivo corrupto"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        return ~df["readable"].astype(bool)


@dataclass
class FaceCountRule(QualityRule):
    """Sin cara: descarta. Varias caras: descarta, salvo que la mayor sea
    dominante (área >= dominance_ratio veces la segunda), en cuyo caso se usa esa."""

    dominance_ratio: float = 2.0
    name: str = "cero o varias caras"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        no_face = df["n_faces"] == 0
        ambiguous = (df["n_faces"] > 1) & (df["face_area_ratio"] < self.dominance_ratio)
        return no_face | ambiguous

    def describe(self) -> str:
        return f"{self.name} (mayor/segunda < {self.dominance_ratio:g})"


@dataclass
class MinFaceSizeRule(QualityRule):
    min_side: int = 40
    name: str = "cara chica"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        return df["face_min_side"] < self.min_side

    def describe(self) -> str:
        return f"{self.name} (lado < {self.min_side} px)"


@dataclass
class BlurRule(QualityRule):
    min_laplacian_var: float = 30.0
    name: str = "borrosa"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        return df["laplacian_var"] < self.min_laplacian_var

    def describe(self) -> str:
        return f"{self.name} (var. Laplaciano < {self.min_laplacian_var:g})"


@dataclass
class ExposureRule(QualityRule):
    min_brightness: float = 40.0
    max_brightness: float = 215.0
    max_clipped_fraction: float = 0.3
    name: str = "exposición extrema"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        clipped = df[["dark_fraction", "bright_fraction"]].max(axis=1)
        return (
            (df["brightness"] < self.min_brightness)
            | (df["brightness"] > self.max_brightness)
            | (clipped > self.max_clipped_fraction)
        )

    def describe(self) -> str:
        return (
            f"{self.name} (brillo fuera de [{self.min_brightness:g}, {self.max_brightness:g}] "
            f"o > {self.max_clipped_fraction:.0%} de píxeles saturados)"
        )


@dataclass
class FlagRule(QualityRule):
    """Descarta las filas marcadas en una columna booleana calculada aparte (por ejemplo, duplicados)."""

    column: str = "is_duplicate"
    name: str = "duplicado"

    def fails(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].fillna(False).astype(bool)


class QualityFilter:
    """Aplica las reglas en orden. Cada imagen descartada se atribuye a la primera regla que falla,
    así la suma de descartes por regla coincide con el total."""

    def __init__(self, rules: list[QualityRule]) -> None:
        self.rules = rules

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devuelve (df con columna discard_reason, reporte por regla)."""
        out = df.copy()
        out["discard_reason"] = None
        report = []
        for rule in self.rules:
            pending = out["discard_reason"].isna()
            # Las filas ya descartadas pueden tener NaN en las columnas de reglas posteriores.
            failing = pending & rule.fails(out).fillna(False).astype(bool)
            out.loc[failing, "discard_reason"] = rule.name
            report.append({"regla": rule.describe(), "descartadas": int(failing.sum()), "quedan": int((pending & ~failing).sum())})
        return out, pd.DataFrame(report)
