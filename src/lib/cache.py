"""Caché en disco para pasos lentos de las notebooks.

El archivo se nombra con una clave (fingerprint) calculada a partir de todo lo que define el resultado:
datos de entrada, parámetros y código. Si cambia algo, cambia la clave y se recalcula solo; los archivos
viejos quedan en la carpeta y se pueden borrar sin riesgo.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Codec:
    """Cómo se guarda y se lee un tipo de resultado."""

    suffix: str
    save: Callable[[Any, Path], None]
    load: Callable[[Path], Any]


PANDAS = Codec(".pkl", lambda obj, path: obj.to_pickle(path), pd.read_pickle)   # conserva índice y tipos
ARRAY = Codec(".npy", lambda obj, path: np.save(path, obj), np.load)


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _source_digest(obj: Any) -> str | None:
    try:
        return _digest(inspect.getsource(obj).encode("utf-8"))
    except (OSError, TypeError):
        return None


def _normalize(value: Any) -> Any:
    """Convierte un valor en algo serializable que cambia si el valor cambia."""
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": str(value.dtype), "sha1": _digest(np.ascontiguousarray(value).tobytes())}
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return {"pandas": _digest(pd.util.hash_pandas_object(value).to_numpy().tobytes())}
    if isinstance(value, ModuleType):
        return {"module": value.__name__, "sha1": _source_digest(value)}
    if callable(value):
        # Se usa el código de todo el módulo de la función: también cubre las funciones auxiliares que llama.
        # Una función definida en la notebook no tiene módulo con archivo: se usa solo su código.
        module = sys.modules.get(getattr(value, "__module__", ""), None)
        source = _source_digest(module) if module is not None else None
        return {"callable": getattr(value, "__qualname__", repr(value)), "sha1": source or _source_digest(value)}
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def fingerprint(**parts: Any) -> str:
    """Clave corta para nombrar una caché a partir de datos, parámetros y funciones."""
    payload = json.dumps(_normalize(parts), sort_keys=True, default=str)
    return _digest(payload.encode("utf-8"))[:12]


def cached(directory: Path, name: str, key: str, compute: Callable[[], Any], codec: Codec) -> Any:
    """Devuelve el resultado guardado en directory/<name>_<key><suffix>; si no existe, lo calcula y lo guarda."""
    path = Path(directory) / f"{name}_{key}{codec.suffix}"
    if path.exists():
        return codec.load(path)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    codec.save(result, path)
    return result
