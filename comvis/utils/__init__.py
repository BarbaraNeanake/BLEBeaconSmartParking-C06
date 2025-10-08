"""
SPARK Car Detection Utils Package
Modular utilities for YOLOv2-ResNet car detection pipeline
"""

from .model import YOLOv2ResNet
from .loss import CIoUYOLOLoss
from .data_utils import ParsedYOLODataset, compute_anchors
from .train_utils import ModelTrainer
from .inference_utils import ModelInference
from .config import Config

__all__ = [
    'YOLOv2ResNet',
    'CIoUYOLOLoss', 
    'ParsedYOLODataset',
    'compute_anchors',
    'ModelTrainer',
    'ModelInference',
    'Config'
]