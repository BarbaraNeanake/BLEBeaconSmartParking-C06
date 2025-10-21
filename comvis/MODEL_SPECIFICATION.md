# SPARK Car Detection Model Specification

## Overview

**Model Name**: YOLOv2-ResNet34  
**Task**: Single-class object detection (cars)  
**Framework**: PyTorch  
**Version**: 1.0  
**Last Updated**: October 20, 2025

---

## Architecture

### Backbone: ResNet34
- **Type**: Convolutional Neural Network (ResNet-34)
- **Pre-training**: ImageNet weights
- **Output Channels**: 512
- **Feature Map Size**: 13×13 (for 416×416 input)
- **Parameters**: ~21.3M (backbone only)

### Detection Head: YOLOv2-style
```
Input: 512 channels from ResNet34 backbone

Layer 1: Conv2d(512 → 512, kernel=3×3, padding=1)
         BatchNorm2d(512)
         LeakyReLU(0.1)
         Dropout2d(0.2)

Layer 2: Conv2d(512 → 512, kernel=3×3, padding=1)
         BatchNorm2d(512)
         LeakyReLU(0.1)

Layer 3: Conv2d(512 → 256, kernel=1×1)
         BatchNorm2d(256)
         LeakyReLU(0.1)

Layer 4: Conv2d(256 → 30, kernel=1×1)  # 5 anchors × (5 + 1 class)

Output: 30 channels
```

### Total Parameters
- **Backbone**: ~21,300,000 parameters
- **Detection Head**: ~1,500,000 parameters
- **Total**: ~22,800,000 parameters

---

## Input Specifications

### Image Input
- **Format**: RGB images
- **Input Size**: 416 × 416 pixels
- **Normalization**: ImageNet statistics
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Data Type**: `float32`
- **Value Range**: [0, 1] before normalization

### Preprocessing Pipeline
```python
1. Resize image to 416×416 (maintain aspect ratio with padding)
2. Convert to RGB if grayscale
3. Normalize using ImageNet statistics
4. Convert to tensor: (H, W, C) → (C, H, W)
5. Add batch dimension: (C, H, W) → (1, C, H, W)
```

---

## Output Specifications

### Raw Output
- **Shape**: `(batch_size, 30, 13, 13)`
- **Grid Size**: 13 × 13
- **Anchors per Grid Cell**: 5
- **Channels per Anchor**: 6 (tx, ty, tw, th, confidence, class_prob)

### Output Tensor Structure
```
For each anchor at grid cell (i, j):
- Channel 0-4:   Anchor 1 [tx, ty, tw, th, obj_conf, class_prob]
- Channel 5-9:   Anchor 2 [tx, ty, tw, th, obj_conf, class_prob]
- Channel 10-14: Anchor 3 [tx, ty, tw, th, obj_conf, class_prob]
- Channel 15-19: Anchor 4 [tx, ty, tw, th, obj_conf, class_prob]
- Channel 20-24: Anchor 5 [tx, ty, tw, th, obj_conf, class_prob]
- Channel 25-29: Remaining channels
```

### Anchor Boxes
- **Number of Anchors**: 5
- **Anchor Dimensions** (width, height in pixels):
  ```
  Computed via K-means clustering on training dataset
  Stored in: datasets/COCO_car/anchors.npy
  ```

### Detection Output (Post-processed)
```python
{
    'boxes': [
        [x_min, y_min, x_max, y_max],  # Bounding box coordinates
        ...
    ],
    'scores': [0.95, 0.87, ...],        # Confidence scores
    'labels': [0, 0, ...]                # Class labels (always 0 for car)
}
```

---

## Inference Parameters

### Thresholds
- **Confidence Threshold**: 0.5 (configurable: 0.4 - 0.7)
- **NMS Threshold**: 0.45 (configurable: 0.3 - 0.5)
- **Max Detections per Image**: 100

### Post-processing Steps
1. **Sigmoid activation** on tx, ty, confidence, class_prob
2. **Exponential transform** on tw, th (with clipping to prevent overflow)
3. **Decode bounding boxes** using anchor boxes and grid offsets
4. **Filter by confidence** threshold
5. **Non-Maximum Suppression (NMS)** to remove duplicate detections
6. **Convert coordinates** from grid space to image space

---

## Performance Metrics

### Target Performance
- **mAP@0.5**: > 0.70
- **Inference Time**: < 50ms per image (GPU)
- **Inference Time**: < 200ms per image (CPU)

### Hardware Requirements

#### Training
- **GPU**: NVIDIA GPU with 6GB+ VRAM (CUDA compatible)
- **RAM**: 16GB recommended
- **Storage**: 5GB for dataset + models

#### Inference (Deployment)
- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with 2GB+ VRAM
- **Storage**: 100MB for model weights

---

## Model Files

### Saved Model Format
- **File Format**: PyTorch state_dict (`.pth`)
- **Model Weights**: `models/best_model.pth`
- **File Size**: ~90 MB

### Loading Model
```python
import torch
from utils.model import YOLOv2ResNet

# Initialize model
model = YOLOv2ResNet(
    num_anchors=5,
    num_classes=1,
    pretrained=False  # Set True only for training
)

# Load weights
model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
model.eval()

# Move to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

---

## Deployment Considerations

### API Input/Output

#### REST API Request
```json
{
    "image": "base64_encoded_image_string",
    "confidence_threshold": 0.5,  // optional
    "nms_threshold": 0.45         // optional
}
```

#### REST API Response
```json
{
    "detections": [
        {
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": 0.95,
            "class": "car"
        }
    ],
    "num_detections": 3,
    "inference_time_ms": 45.2
}
```

### Optimization Options

#### 1. **Model Quantization**
- Convert to INT8 for 4x size reduction
- Slight accuracy trade-off (~2-3% mAP drop)
- 2-3x inference speedup on CPU

#### 2. **ONNX Export**
```python
import torch.onnx

dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

#### 3. **TorchScript**
```python
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

#### 4. **TensorRT** (NVIDIA GPUs)
- Convert ONNX → TensorRT for maximum GPU performance
- 5-10x speedup possible on NVIDIA hardware

---

## Dependencies

### Core Libraries
```
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
pillow >= 8.3.0
opencv-python >= 4.5.0  # Optional for video processing
```

### Deployment-Specific
```
# REST API
flask >= 2.0.0
# or
fastapi >= 0.70.0
uvicorn >= 0.15.0

# Image processing
pillow >= 8.3.0

# Optional optimizations
onnx >= 1.10.0
onnxruntime >= 1.10.0
```

---

## Training Configuration

### Hyperparameters
```python
num_epochs: 50
batch_size: 8
learning_rate: 0.001
weight_decay: 1e-4
optimizer: Adam
```

### Loss Function
Custom YOLOv2 loss with:
- **Coordinate Loss** (λ=5.0): MSE on bounding box coordinates
- **Objectness Loss** (λ=1.0): BCE on confidence scores
- **No-object Loss** (λ=0.5): BCE on background predictions
- **Classification Loss**: BCE on class probabilities

### Data Augmentation
- Random horizontal flip
- Random brightness/contrast adjustment
- Random scaling (0.8 - 1.2)
- Random cropping

---

## Limitations & Considerations

### Known Limitations
1. **Single Class Only**: Currently trained only for car detection
2. **Fixed Input Size**: Requires 416×416 input (can be modified)
3. **Grid-based**: May struggle with very small objects (<20 pixels)
4. **Occlusion**: Performance degrades with heavy occlusion

### Best Use Cases
- ✅ Parking lot monitoring
- ✅ Traffic surveillance
- ✅ Vehicle counting
- ✅ Real-time applications (with GPU)

### Not Recommended For
- ❌ Multi-class vehicle detection (cars, trucks, motorcycles, etc.)
- ❌ License plate detection (too small)
- ❌ Night-time detection (without IR cameras)
- ❌ Extreme weather conditions

---

## Version History

### v1.0 (October 2025)
- Initial release with ResNet34 backbone
- Migrated from ResNet50 for improved efficiency
- Optimized for proof-of-concept deployment

---

## Contact & Support

**Project**: SPARK - Smart Parking System  
**Repository**: BLEBeaconSmartParking-C06  
**Branch**: Danish  

For deployment questions, refer to:
- `deployment/README.md` - Deployment guide
- `comvis/FIXES_SUMMARY.md` - Quick reference
- `comvis/utils/config.py` - Configuration options

---

## License & Usage

This model is part of the SPARK Smart Parking System project.
Ensure compliance with data privacy regulations when deploying in production.
