# Pure Python Inference Engine for YOLOv2-ResNet34

Complete rewrite of the Cython engine in pure Python using NumPy. No compilation needed!

## Features

✅ **Pure Python** - No Cython, no C++ build tools needed  
✅ **NumPy Based** - All operations use NumPy for performance  
✅ **PyTorch Compatible** - Load weights directly from .pth files  
✅ **Easy to Deploy** - Minimal dependencies, works on any platform  

## Installation

```bash
cd e:\projects\SPARK\deployment
pip install -r requirements.txt
```

## Project Structure

```
inference_engine/
├── __init__.py           # Public API
├── math_ops.py          # NumPy operations (conv2d, activations, BN, pooling)
├── nms.py               # Non-Maximum Suppression (IoU, CIoU)
├── model.py             # ResNet34 backbone + YOLOv2 head
└── inference.py         # Main inference engine
```

## Usage

### Basic Inference

```python
from inference_engine import create_inference_engine

# Create engine
engine = create_inference_engine(
    model_path="path/to/model.pth",
    config_path="config.json",
    input_size=416,
    backend="resnet34"
)

# Run inference
import cv2
image = cv2.imread("image.jpg")
detections = engine.predict(image)

# Print results
for det in detections:
    print(f"Class: {det['class_name']}")
    print(f"Confidence: {det['confidence']:.2f}")
    print(f"BBox: {det['bbox']}")
```

### Configuration File

Create `config.json`:

```json
{
  "num_classes": 1,
  "conf_threshold": 0.5,
  "nms_threshold": 0.4,
  "class_names": ["car"],
  "anchors": [
    [0.57273, 0.677385],
    [1.87446, 2.06253],
    [3.33843, 5.47434],
    [7.88282, 3.52778],
    [9.77052, 9.16828]
  ]
}
```

## Architecture

### ResNet34 Backbone
- Input: (3, 416, 416)
- Layers: 7×7 conv + 3 sets of residual blocks [3, 4, 6, 3]
- Output: (512, 13, 13)

### YOLOv2 Detection Head
- Input: (512, 13, 13)
- 512 → 512 → 256 → 30 channels (5 anchors × (5 + num_classes))
- Output format: [tx, ty, tw, th, confidence, class_probs]

## Key Modules

### math_ops.py
- `conv2d()` - 2D convolution
- `relu()`, `sigmoid()` - Activation functions
- `batch_norm()` - Batch normalization
- `max_pool2d()` - Max pooling
- `adaptive_avg_pool2d()` - Adaptive average pooling

### nms.py
- `nms()` - Standard Non-Maximum Suppression
- `soft_nms()` - Soft NMS with Gaussian weighting
- `compute_iou()` - Intersection over Union
- `compute_ciou()` - Complete IoU

### model.py
- `ResBlock` - Single residual block
- `ResNetBackbone` - Full ResNet34 backbone
- `YOLODetectionHead` - YOLO prediction head
- `YOLOv2ResNet` - Complete model

### inference.py
- `InferenceEngine` - Main inference class
- `create_inference_engine()` - Factory function

## Performance

Expected inference time (single image, 416×416):
- Backbone: ~50-100ms
- Detection Head: ~10-20ms
- NMS: ~5-10ms
- **Total: ~100-150ms**

## Migration from Cython

The old Cython modules (`_math.pyx`, `_nms.pyx`, `_model.pyx`, `core.pyx`) have been completely replaced:

| Old File | New File | Status |
|----------|----------|--------|
| `_math.pyx` | `math_ops.py` | ✅ Replaced |
| `_nms.pyx` | `nms.py` | ✅ Replaced |
| `_model.pyx` | `model.py` | ✅ Replaced |
| `core.pyx` | `inference.py` | ✅ Replaced |

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install torch
```

### ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

### Weight loading fails
Ensure the `.pth` file is compatible with PyTorch and config.json matches the model.

## Notes

- All operations use float32 for compatibility
- Inference is CPU-based (GPU support can be added if needed)
- Model weights are loaded on CPU and kept in memory
- Image preprocessing includes resizing and padding to 416×416

## Future Enhancements

- [ ] GPU support (CUDA/Metal)
- [ ] Batch inference optimization
- [ ] Model quantization
- [ ] ONNX export support
- [ ] Performance profiling
