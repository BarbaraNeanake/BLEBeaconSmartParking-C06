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
            """Try to find key in state_dict"""
            full_key = f"{prefix}{key_name}"
            if full_key in state_dict:
                value = state_dict[full_key]
                # Convert torch tensor to numpy if needed
                if hasattr(value, 'numpy'):
                    return value.numpy()
                return value
            raise KeyError(f"Key '{full_key}' not found in state_dict")
        
        try:
            self.conv1_weight = get_key("conv1.weight")
            self.bn1_weight = get_key("bn1.weight")
            self.bn1_bias = get_key("bn1.bias")
            self.bn1_mean = get_key("bn1.running_mean")
            self.bn1_var = get_key("bn1.running_var")
            
            self.conv2_weight = get_key("conv2.weight")
            self.bn2_weight = get_key("bn2.weight")
            self.bn2_bias = get_key("bn2.bias")
            self.bn2_mean = get_key("bn2.running_mean")
            self.bn2_var = get_key("bn2.running_var")
            
            if self.downsample_weight is not None:
                self.downsample_weight = get_key("downsample.0.weight")
                self.downsample_bn_weight = get_key("downsample.1.weight")
                self.downsample_bn_bias = get_key("downsample.1.bias")
                self.downsample_bn_mean = get_key("downsample.1.running_mean")
                self.downsample_bn_var = get_key("downsample.1.running_var")
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
            """Get key and convert to numpy if needed"""
            if key_name in state_dict:
                value = state_dict[key_name]
                if hasattr(value, 'numpy'):
                    return value.numpy()
                return value
            raise KeyError(f"Key '{key_name}' not found")
        
        try:
            # Load initial conv (backbone.0 and backbone.1)
            self.conv1_weight = get_key("backbone.0.weight")
            self.bn1_weight = get_key("backbone.1.weight")
            self.bn1_bias = get_key("backbone.1.bias")
            self.bn1_mean = get_key("backbone.1.running_mean")
            self.bn1_var = get_key("backbone.1.running_var")
            
            # Load residual layers
            # backbone.4 = layer1, backbone.5 = layer2, backbone.6 = layer3, backbone.7 = layer4
            for i, block in enumerate(self.layer1_blocks):
                block.load_weights(state_dict, f"backbone.4.{i}.")
            for i, block in enumerate(self.layer2_blocks):
                block.load_weights(state_dict, f"backbone.5.{i}.")
            for i, block in enumerate(self.layer3_blocks):
                block.load_weights(state_dict, f"backbone.6.{i}.")
            for i, block in enumerate(self.layer4_blocks):
                block.load_weights(state_dict, f"backbone.7.{i}.")
                
            print("✓ Backbone weights loaded successfully")
        except Exception as e:
            print(f"Warning during backbone weight loading: {e}")


class YOLODetectionHead:
    """YOLO detection head"""
    
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        
        # Detection head layers: 512 → 512 → 256 → 256 → (B*(5+C))
        # where B=5 anchors, C=num_classes
        self.num_anchors = 5
        
        # Conv layer 1: 512 -> 512
        self.conv1_weight = np.random.randn(512, 512, 1, 1).astype(np.float32) * 0.01
        self.conv1_bn_weight = np.ones(512, dtype=np.float32)
        self.conv1_bn_bias = np.zeros(512, dtype=np.float32)
        self.conv1_bn_mean = np.zeros(512, dtype=np.float32)
        self.conv1_bn_var = np.ones(512, dtype=np.float32)
        
        # Conv layer 2: 512 -> 256
        self.conv2_weight = np.random.randn(256, 512, 1, 1).astype(np.float32) * 0.01
        self.conv2_bn_weight = np.ones(256, dtype=np.float32)
        self.conv2_bn_bias = np.zeros(256, dtype=np.float32)
        self.conv2_bn_mean = np.zeros(256, dtype=np.float32)
        self.conv2_bn_var = np.ones(256, dtype=np.float32)
        
        # Conv layer 3: 256 -> 256
        self.conv3_weight = np.random.randn(256, 256, 1, 1).astype(np.float32) * 0.01
        self.conv3_bn_weight = np.ones(256, dtype=np.float32)
        self.conv3_bn_bias = np.zeros(256, dtype=np.float32)
        self.conv3_bn_mean = np.zeros(256, dtype=np.float32)
        self.conv3_bn_var = np.ones(256, dtype=np.float32)
        
        # Final prediction layer: 256 -> (B*(5+C))
        out_channels = self.num_anchors * (5 + num_classes)
        self.conv_pred_weight = np.random.randn(out_channels, 256, 1, 1).astype(np.float32) * 0.01
        self.conv_pred_bias = np.zeros(out_channels, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through detection head"""
        # Conv layer 1 with BN + ReLU
        x = conv2d(x, self.conv1_weight, stride=1)
        x = batch_norm(x, self.conv1_bn_weight, self.conv1_bn_bias, 
                      self.conv1_bn_mean, self.conv1_bn_var)
        x = relu(x)
        
        # Conv layer 2 with BN + ReLU
        x = conv2d(x, self.conv2_weight, stride=1)
        x = batch_norm(x, self.conv2_bn_weight, self.conv2_bn_bias,
                      self.conv2_bn_mean, self.conv2_bn_var)
        x = relu(x)
        
        # Conv layer 3 with BN + ReLU
        x = conv2d(x, self.conv3_weight, stride=1)
        x = batch_norm(x, self.conv3_bn_weight, self.conv3_bn_bias,
                      self.conv3_bn_mean, self.conv3_bn_var)
        x = relu(x)
        
        # Prediction layer (no activation)
        x = conv2d(x, self.conv_pred_weight, self.conv_pred_bias, stride=1)
        
        return x
    
    def load_weights(self, state_dict: Dict):
        """Load detection head weights"""
        def get_key(key_name):
            """Get key and convert to numpy if needed"""
            if key_name in state_dict:
                value = state_dict[key_name]
                if hasattr(value, 'numpy'):
                    return value.numpy()
                return value
            raise KeyError(f"Key '{key_name}' not found")
        
        try:
            # conv.0 and conv.1 = first conv layer + bn (512 -> 512)
            self.conv1_weight = get_key("conv.0.weight")
            self.conv1_bn_weight = get_key("conv.1.weight")
            self.conv1_bn_bias = get_key("conv.1.bias")
            self.conv1_bn_mean = get_key("conv.1.running_mean")
            self.conv1_bn_var = get_key("conv.1.running_var")
            
            # conv.4 and conv.5 = second conv layer + bn (512 -> 256)
            self.conv2_weight = get_key("conv.4.weight")
            self.conv2_bn_weight = get_key("conv.5.weight")
            self.conv2_bn_bias = get_key("conv.5.bias")
            self.conv2_bn_mean = get_key("conv.5.running_mean")
            self.conv2_bn_var = get_key("conv.5.running_var")
            
            # conv.7 and conv.8 = third conv layer + bn (256 -> 256)
            self.conv3_weight = get_key("conv.7.weight")
            self.conv3_bn_weight = get_key("conv.8.weight")
            self.conv3_bn_bias = get_key("conv.8.bias")
            self.conv3_bn_mean = get_key("conv.8.running_mean")
            self.conv3_bn_var = get_key("conv.8.running_var")
            
            # conv.10 = final prediction layer (256 -> out_channels)
            self.conv_pred_weight = get_key("conv.10.weight")
            self.conv_pred_bias = get_key("conv.10.bias")
            
            print("✓ Detection head weights loaded successfully")
        except KeyError as e:
            print(f"Warning: Could not load detection head weight {e}")


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
        print(f"Loading weights from checkpoint with {len(state_dict)} keys")
        
        # Load backbone weights (all keys starting with 'backbone.')
        self.backbone.load_weights(state_dict)
        
        # Load detection head weights (all keys starting with 'conv.')
        self.detection_head.load_weights(state_dict)
        
        print("✓ Model weights loaded successfully")
