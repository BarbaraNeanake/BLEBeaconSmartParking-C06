"""
SPARK Car Detection Utils Package
Modular utilities for YOLOv2-ResNet car detection pipeline
"""

from .model import YOLOv2ResNet
from .loss import CIoUYOLOLoss
from .data_utils import ParsedYOLODataset, compute_anchors
from .train_utils import ModelTrainer, train_model
from .inference_utils import ModelInference, run_inference
from .config import Config, default_config

__all__ = [
    'YOLOv2ResNet',
    'CIoUYOLOLoss', 
    'ParsedYOLODataset',
    'compute_anchors',
    'ModelTrainer',
    'train_model',
    'ModelInference',
    'run_inference',
    'Config',
    'default_config'
]