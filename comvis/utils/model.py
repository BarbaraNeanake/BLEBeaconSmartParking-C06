"""
Model definitions for SPARK car detection pipeline
"""
import torch
import torch.nn as nn
from torchvision.models import resnet34
from typing import Optional


class YOLOv2ResNet(nn.Module):
    """
    YOLOv2-style detection head on ResNet34 backbone
    Lighter architecture optimized for proof-of-concept
    """
    
    def __init__(self, num_anchors: int = 5, num_classes: int = 1, pretrained: bool = True):
        super(YOLOv2ResNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Load pre-trained ResNet34 backbone
        self.backbone = resnet34(pretrained=pretrained)
        # Remove the final fully connected layer and adaptive pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Improved YOLOv2 detection head with feature refinement
        self.conv = nn.Sequential(
            # First refinement layer (ResNet34 outputs 512 channels, not 2048)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),  # Regularization
            
            # Second refinement layer
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Dimensionality reduction
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Final prediction layer
            nn.Conv2d(256, num_anchors * (5 + num_classes), kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize detection head weights"""
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        x = self.backbone(x)
        x = self.conv(x)
        return x
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for head-only training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for full model training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_params(self) -> tuple:
        """Get number of parameters in backbone and head separately"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.conv.parameters())
        return backbone_params, head_params
    
    def load_pretrained_weights(self, model_path: str, device: str = 'cpu') -> None:
        """Load pretrained weights"""
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.load_state_dict(state_dict)
            print(f"✓ Loaded model weights from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load weights from {model_path}: {e}")
            raise
    
    def save_weights(self, model_path: str) -> None:
        """Save model weights"""
        try:
            torch.save(self.state_dict(), model_path)
            print(f"✓ Saved model weights to {model_path}")
        except Exception as e:
            print(f"✗ Failed to save weights to {model_path}: {e}")
            raise


def create_model(config, device: str = 'cpu', pretrained: bool = True) -> YOLOv2ResNet:
    """
    Factory function to create YOLOv2ResNet model with configuration
    
    Args:
        config: Configuration object
        device: Target device ('cpu' or 'cuda')
        pretrained: Whether to use pretrained ResNet50 backbone
        
    Returns:
        YOLOv2ResNet model instance
    """
    model = YOLOv2ResNet(
        num_anchors=config.num_anchors,
        num_classes=config.num_classes,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    # Print model info
    backbone_params, head_params = model.get_num_params()
    total_params = backbone_params + head_params
    
    print(f"Model created:")
    print(f"  - Backbone parameters: {backbone_params:,}")
    print(f"  - Detection head parameters: {head_params:,}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Device: {device}")
    
    return model