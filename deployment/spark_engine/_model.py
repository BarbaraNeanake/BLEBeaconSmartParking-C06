"""
Pure Python ResNet34 and YOLOv2 model implementations using NumPy
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from ._math import conv2d, relu, batch_norm, max_pool2d, adaptive_avg_pool2d


class ResBlock:
    """ResNet Basic Block (identity residual block)"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, eps: float = 1e-5):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.eps = eps
        
        # Weights
        self.conv1_weight = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.01
        self.bn1_weight = np.ones(out_channels, dtype=np.float32)
        self.bn1_bias = np.zeros(out_channels, dtype=np.float32)
        self.bn1_mean = np.zeros(out_channels, dtype=np.float32)
        self.bn1_var = np.ones(out_channels, dtype=np.float32)
        
        self.conv2_weight = np.random.randn(out_channels, out_channels, 3, 3).astype(np.float32) * 0.01
        self.bn2_weight = np.ones(out_channels, dtype=np.float32)
        self.bn2_bias = np.zeros(out_channels, dtype=np.float32)
        self.bn2_mean = np.zeros(out_channels, dtype=np.float32)
        self.bn2_var = np.ones(out_channels, dtype=np.float32)
        
        # Shortcut (downsample)
        if stride != 1 or in_channels != out_channels:
            self.downsample_weight = np.random.randn(out_channels, in_channels, 1, 1).astype(np.float32) * 0.01
            self.downsample_bn_weight = np.ones(out_channels, dtype=np.float32)
            self.downsample_bn_bias = np.zeros(out_channels, dtype=np.float32)
            self.downsample_bn_mean = np.zeros(out_channels, dtype=np.float32)
            self.downsample_bn_var = np.ones(out_channels, dtype=np.float32)
        else:
            self.downsample_weight = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        identity = x
        
        # Conv1 - BN1 - ReLU
        out = conv2d(x, self.conv1_weight, stride=self.stride, pad=1)
        out = batch_norm(out, self.bn1_weight, self.bn1_bias, self.bn1_mean, self.bn1_var, self.eps)
        out = relu(out)
        
        # Conv2 - BN2
        out = conv2d(out, self.conv2_weight, stride=1, pad=1)
        out = batch_norm(out, self.bn2_weight, self.bn2_bias, self.bn2_mean, self.bn2_var, self.eps)
        
        # Shortcut
        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = conv2d(identity, self.downsample_weight, stride=self.stride)
            identity = batch_norm(identity, self.downsample_bn_weight, self.downsample_bn_bias,
                                 self.downsample_bn_mean, self.downsample_bn_var, self.eps)
        
        # Add and ReLU
        out = out + identity
        out = relu(out)
        
        return out
    
    def load_weights(self, state_dict: Dict, prefix: str = ""):
        """Load weights from PyTorch state dict"""
        def get_key(key_name):
            """Try to find key with or without backbone prefix"""
            if f"{prefix}{key_name}" in state_dict:
                return state_dict[f"{prefix}{key_name}"]
            elif f"{prefix}backbone.{key_name}" in state_dict:
                return state_dict[f"{prefix}backbone.{key_name}"]
            else:
                raise KeyError(f"Key '{key_name}' not found in state_dict")
        
        try:
            self.conv1_weight = get_key("conv1.weight").numpy()
            self.bn1_weight = get_key("bn1.weight").numpy()
            self.bn1_bias = get_key("bn1.bias").numpy()
            self.bn1_mean = get_key("bn1.running_mean").numpy()
            self.bn1_var = get_key("bn1.running_var").numpy()
            
            self.conv2_weight = get_key("conv2.weight").numpy()
            self.bn2_weight = get_key("bn2.weight").numpy()
            self.bn2_bias = get_key("bn2.bias").numpy()
            self.bn2_mean = get_key("bn2.running_mean").numpy()
            self.bn2_var = get_key("bn2.running_var").numpy()
            
            if self.downsample_weight is not None:
                self.downsample_weight = get_key("downsample.0.weight").numpy()
                self.downsample_bn_weight = get_key("downsample.1.weight").numpy()
                self.downsample_bn_bias = get_key("downsample.1.bias").numpy()
                self.downsample_bn_mean = get_key("downsample.1.running_mean").numpy()
                self.downsample_bn_var = get_key("downsample.1.running_var").numpy()
        except KeyError as e:
            print(f"Warning: Could not load weight {e}")


class ResNetBackbone:
    """ResNet-34 backbone"""
    
    def __init__(self, layers: List[int] = None, num_classes: int = 1000, eps: float = 1e-5):
        if layers is None:
            layers = [3, 4, 6, 3]  # ResNet34
        
        self.layers = layers
        self.num_classes = num_classes
        self.eps = eps
        
        # Initial convolution
        self.conv1_weight = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
        self.bn1_weight = np.ones(64, dtype=np.float32)
        self.bn1_bias = np.zeros(64, dtype=np.float32)
        self.bn1_mean = np.zeros(64, dtype=np.float32)
        self.bn1_var = np.ones(64, dtype=np.float32)
        
        # Build residual layers
        self.layer1_blocks = [ResBlock(64, 64, stride=1) for _ in range(layers[0])]
        self.layer2_blocks = [ResBlock(64, 128, stride=2)] + [ResBlock(128, 128, stride=1) for _ in range(layers[1] - 1)]
        self.layer3_blocks = [ResBlock(128, 256, stride=2)] + [ResBlock(256, 256, stride=1) for _ in range(layers[2] - 1)]
        self.layer4_blocks = [ResBlock(256, 512, stride=2)] + [ResBlock(512, 512, stride=1) for _ in range(layers[3] - 1)]
        
        # Classification head
        self.fc_weight = np.random.randn(num_classes, 512).astype(np.float32) * 0.01
        self.fc_bias = np.zeros(num_classes, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through backbone"""
        # Initial layers
        x = conv2d(x, self.conv1_weight, stride=2, pad=3)
        x = batch_norm(x, self.bn1_weight, self.bn1_bias, self.bn1_mean, self.bn1_var, self.eps)
        x = relu(x)
        x = max_pool2d(x, 3, stride=2, padding=1)
        
        # Residual layers
        for block in self.layer1_blocks:
            x = block.forward(x)
        for block in self.layer2_blocks:
            x = block.forward(x)
        for block in self.layer3_blocks:
            x = block.forward(x)
        for block in self.layer4_blocks:
            x = block.forward(x)
        
        return x
    
    def load_weights(self, state_dict: Dict):
        """Load weights from PyTorch state dict"""
        def get_key(key_name):
            """Try to find key with or without prefix"""
            if key_name in state_dict:
                return state_dict[key_name]
            backbone_key = f"backbone.{key_name}"
            if backbone_key in state_dict:
                return state_dict[backbone_key]
            raise KeyError(f"Key '{key_name}' not found")
        
        try:
            # Load initial conv
            self.conv1_weight = get_key("conv1.weight").numpy()
            self.bn1_weight = get_key("bn1.weight").numpy()
            self.bn1_bias = get_key("bn1.bias").numpy()
            self.bn1_mean = get_key("bn1.running_mean").numpy()
            self.bn1_var = get_key("bn1.running_var").numpy()
            
            # Load residual layers
            for i, block in enumerate(self.layer1_blocks):
                block.load_weights(state_dict, f"layer1.{i}.")
            for i, block in enumerate(self.layer2_blocks):
                block.load_weights(state_dict, f"layer2.{i}.")
            for i, block in enumerate(self.layer3_blocks):
                block.load_weights(state_dict, f"layer3.{i}.")
            for i, block in enumerate(self.layer4_blocks):
                block.load_weights(state_dict, f"layer4.{i}.")
        except Exception as e:
            print(f"Warning during weight loading: {e}")


class YOLODetectionHead:
    """YOLO detection head"""
    
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        
        # Detection head layers: 512 → 512 → 256 → (B*(5+C))
        # where B=5 anchors, C=num_classes
        self.num_anchors = 5
        
        # Conv layers
        self.conv1_weight = np.random.randn(512, 512, 1, 1).astype(np.float32) * 0.01
        self.conv1_bn_weight = np.ones(512, dtype=np.float32)
        self.conv1_bn_bias = np.zeros(512, dtype=np.float32)
        
        self.conv2_weight = np.random.randn(256, 512, 1, 1).astype(np.float32) * 0.01
        self.conv2_bn_weight = np.ones(256, dtype=np.float32)
        self.conv2_bn_bias = np.zeros(256, dtype=np.float32)
        
        # Final prediction layer
        out_channels = self.num_anchors * (5 + num_classes)
        self.conv_pred_weight = np.random.randn(out_channels, 256, 1, 1).astype(np.float32) * 0.01
        self.conv_pred_bias = np.zeros(out_channels, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through detection head"""
        # Conv layers with ReLU
        x = conv2d(x, self.conv1_weight, stride=1)
        x = batch_norm(x, self.conv1_bn_weight, self.conv1_bn_bias, 
                      np.zeros(512), np.ones(512))
        x = relu(x)
        
        x = conv2d(x, self.conv2_weight, stride=1)
        x = batch_norm(x, self.conv2_bn_weight, self.conv2_bn_bias,
                      np.zeros(256), np.ones(256))
        x = relu(x)
        
        # Prediction layer
        x = conv2d(x, self.conv_pred_weight, self.conv_pred_bias, stride=1)
        
        return x
    
    def load_weights(self, state_dict: Dict):
        """Load detection head weights"""
        pass  # Implement as needed


class YOLOv2ResNet:
    """Complete YOLOv2 with ResNet34 backbone"""
    
    def __init__(self, num_classes: int = 1, backbone_type: str = "resnet34"):
        self.num_classes = num_classes
        self.backbone = ResNetBackbone([3, 4, 6, 3], num_classes=1000)
        self.detection_head = YOLODetectionHead(num_classes)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        backbone_out = self.backbone.forward(x)
        detections = self.detection_head.forward(backbone_out)
        return detections
    
    def load_weights(self, state_dict: Dict, anchors: np.ndarray = None):
        """Load complete model weights"""
        # Categorize weights
        backbone_weights = {}
        head_weights = {}
        
        for key, value in state_dict.items():
            if any(x in key for x in ["layer1", "layer2", "layer3", "layer4", "conv1", "bn1"]):
                backbone_weights[key.replace("backbone.", "")] = value
            elif "conv." in key:
                head_weights[key] = value
        
        # Load backbone
        if backbone_weights:
            self.backbone.load_weights(backbone_weights)
        
        # Load detection head
        if head_weights:
            self.detection_head.load_weights(head_weights)
