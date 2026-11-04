"""Modelos propios de la Parte 2. Tienen que ser importables desde el backend."""

from lib.models.custom_cnn import CustomCNNConfig, CustomFaceCNN, ResidualBlock, count_parameters
from lib.models.export import export_onnx
from lib.models.heads import ArcFaceHead, SoftmaxHead

__all__ = ["ArcFaceHead", "CustomCNNConfig", "CustomFaceCNN", "ResidualBlock", "SoftmaxHead", "count_parameters", "export_onnx"]
