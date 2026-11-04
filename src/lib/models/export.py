"""Exportación del extractor a ONNX, el formato que carga la app (FaceService._load_model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_NAME = "input"
OUTPUT_NAME = "embedding"


def export_onnx(model: nn.Module, path: str | Path, input_size: int = 112, opset: int = 17,
                atol: float = 1e-4) -> Path:
    """Exporta `model` (N, 3, H, W) -> (N, D) con batch dinámico y verifica que onnxruntime dé lo mismo que torch."""
    import onnxruntime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    sample = torch.randn(4, 3, input_size, input_size)
    torch.onnx.export(
        model, sample, str(path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    with torch.no_grad():
        expected = model(sample).numpy()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {INPUT_NAME: sample.numpy()})[0]
    if not np.allclose(actual, expected, atol=atol):
        raise RuntimeError(f"ONNX difiere de torch: max |diff| = {np.abs(actual - expected).max():.2e}")
    return path
