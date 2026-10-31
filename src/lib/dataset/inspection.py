from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def inspect_file(path: Path) -> dict:
    """Metadatos de un archivo de imagen. readable=False si PIL u OpenCV no pueden abrirlo."""
    info = {"readable": False, "width": np.nan, "height": np.nan, "mode": None, "channels": np.nan, "format": None}
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            info.update(
                width=img.width,
                height=img.height,
                mode=img.mode,
                channels=len(img.getbands()),
                format=img.format,
            )
    except Exception:
        return info
    info["readable"] = cv2.imread(str(path)) is not None
    return info


def inspect_files(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Agrega a df las columnas de inspect_file para cada path (relativo a root)."""
    infos = pd.DataFrame([inspect_file(Path(root) / p) for p in df["path"]], index=df.index)
    out = pd.concat([df, infos], axis=1)
    out["aspect_ratio"] = out["width"] / out["height"]
    return out
