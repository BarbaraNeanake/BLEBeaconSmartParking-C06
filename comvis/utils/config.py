"""
Configuration management for SPARK car detection pipeline
"""
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json


@dataclass
class Config:
    """Configuration class for car detection - Simple PoC"""
    
    # Model parameters
    num_anchors: int = 5
    num_classes: int = 1
    img_size: int = 416
    grid_size: int = 13
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 15
    
    # Loss parameters - Simple balanced settings
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    lambda_obj: float = 1.0
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Inference parameters - Simple PoC settings
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    
    # Paths
    dataset_root: str = "datasets/COCO_car"
    model_save_dir: str = "models"
    results_dir: str = "results"
    test_images_dir: str = "test_images"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.grid_size = self.img_size // 32
        assert self.train_split + self.val_split + self.test_split == 1.0, "Splits must sum to 1.0"
        assert 0 < self.conf_threshold < 1.0, "Confidence threshold must be between 0 and 1"
        assert 0 < self.nms_threshold < 1.0, "NMS threshold must be between 0 and 1"
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_paths(self) -> dict:
        """Get all relevant file paths"""
        paths = {
            'dataset_root': self.dataset_root,
            'parsed_dataset': os.path.join(self.dataset_root, "parsed_dataset"),
            'anchors_file': os.path.join(self.dataset_root, "anchors.npy"),
            'model_save_dir': self.model_save_dir,
            'best_model': os.path.join(self.model_save_dir, "best_model.pth"),
            'final_model': os.path.join(self.model_save_dir, "final_model.pth"),
            'results_dir': self.results_dir,
            'test_images_dir': self.test_images_dir
        }
        
        # Create directories if they don't exist
        for key in ['model_save_dir', 'results_dir']:
            os.makedirs(paths[key], exist_ok=True)
            
        return paths
    
    def print_config(self) -> None:
        """Print current configuration"""
        print("=" * 50)
        print("SPARK Car Detection - Proof of Concept")
        print("=" * 50)
        print(f"Model: YOLOv2-ResNet50")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"NMS threshold: {self.nms_threshold}")
        print("=" * 50)


# Default configuration instance
default_config = Config()