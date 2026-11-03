"""Modelos propios de la Parte 2. Tienen que ser importables desde el backend."""

from lib.models.custom_cnn import CustomCNNConfig, CustomFaceCNN, ResidualBlock, count_parameters
from lib.models.heads import SoftmaxHead

__all__ = ["CustomCNNConfig", "CustomFaceCNN", "ResidualBlock", "SoftmaxHead", "count_parameters"]
