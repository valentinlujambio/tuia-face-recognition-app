from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

LabelParser = Callable[[str], str]


def strip_prefix(prefix: str) -> LabelParser:
    """Parser de etiquetas para carpetas como 'pins_Adriana Lima' → 'Adriana Lima'."""

    def parse(folder_name: str) -> str:
        return folder_name[len(prefix):] if folder_name.startswith(prefix) else folder_name

    return parse


@dataclass(frozen=True)
class FolderDataset:
    """Dataset con una carpeta por identidad: root/<identidad>/<imagen>.

    Es el formato de Pins y el de un dataset propio, así que el resto del
    pipeline no depende del origen de las imágenes.
    """

    root: Path
    label_parser: LabelParser = lambda name: name
    extensions: Iterable[str] = IMAGE_EXTENSIONS

    def to_frame(self) -> pd.DataFrame:
        """Una fila por archivo: path (relativo a root), identity y extensión."""
        root = Path(self.root)
        extensions = {ext.lower() for ext in self.extensions}
        rows = [
            {
                "path": file.relative_to(root).as_posix(),
                "identity": self.label_parser(identity_dir.name),
                "extension": file.suffix.lower(),
            }
            for identity_dir in sorted(p for p in root.iterdir() if p.is_dir())
            for file in sorted(identity_dir.iterdir())
            if file.is_file() and file.suffix.lower() in extensions
        ]
        if not rows:
            raise ValueError(f"No images found under {root}")
        return pd.DataFrame(rows)


def find_dataset_root(base: Path, min_identity_dirs: int = 2) -> Path:
    """Devuelve la primera carpeta (BFS) bajo base que contiene carpetas de identidades con imágenes.

    Útil porque los zips de Kaggle suelen traer una o dos carpetas contenedoras.
    """
    queue = [Path(base)]
    while queue:
        current = queue.pop(0)
        subdirs = sorted(p for p in current.iterdir() if p.is_dir())
        with_images = [
            d for d in subdirs if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file())
        ]
        if len(with_images) >= min_identity_dirs:
            return current
        queue.extend(subdirs)
    raise FileNotFoundError(f"No identity folders found under {base}")

