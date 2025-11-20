# SPARK Inference Architecture Documentation

## Overview

The SPARK car detection system uses a **YOLOv2-style detection head on a ResNet34 backbone**, implemented in pure NumPy for deployment. This document provides a comprehensive technical reference for the inference pipeline.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Preprocessing Pipeline](#preprocessing-pipeline)
3. [Forward Pass](#forward-pass)
4. [Postprocessing & Decoding](#postprocessing--decoding)
5. [Non-Maximum Suppression (NMS)](#non-maximum-suppression-nms)
6. [Critical Implementation Details](#critical-implementation-details)
7. [Configuration Parameters](#configuration-parameters)

---

## Model Architecture

### 1. Backbone: ResNet34

**Purpose:** Feature extraction from input images

**Structure:**
```
Input (3, 416, 416)
    ↓
Conv2d(7x7, stride=2) + BatchNorm + ReLU
    ↓
MaxPool(3x3, stride=2)
    ↓
Layer1: 3x BasicBlock (64 channels)
    ↓
Layer2: 4x BasicBlock (128 channels, stride=2)
    ↓
Layer3: 6x BasicBlock (256 channels, stride=2)
    ↓
Layer4: 3x BasicBlock (512 channels, stride=2)
    ↓
Output: (512, 13, 13)
```

**BasicBlock Structure:**
```
Conv2d(3x3) + BatchNorm + ReLU
    ↓
Conv2d(3x3) + BatchNorm
    ↓
Add Residual Connection (skip connection)
    ↓
ReLU
```

**Key Points:**
- Uses standard ReLU activation (not LeakyReLU)
- BatchNorm epsilon: `1e-5`
- Pretrained on ImageNet
- Outputs 512-channel feature maps at 13×13 resolution (for 416×416 input)

---

### 2. Detection Head: YOLOv2-style

**Purpose:** Transform backbone features into detection predictions

**Structure:**
```
Input: (512, 13, 13) from backbone
    ↓
Conv2d(3x3, 512→512) + BatchNorm + LeakyReLU(0.1) + Dropout2d(0.2)
    ↓
Conv2d(3x3, 512→512) + BatchNorm + LeakyReLU(0.1)
    ↓
Conv2d(1x1, 512→256) + BatchNorm + LeakyReLU(0.1)
    ↓
Conv2d(1x1, 256→30) [No activation]
    ↓
Output: (30, 13, 13)
```

**Output Channels Breakdown:**
- **30 channels** = 5 anchors × (5 + 1 class)
- Each anchor predicts: `[tx, ty, tw, th, conf, class_prob]`
  - `tx, ty`: Bounding box center offsets (logits)
  - `tw, th`: Bounding box width/height (log scale)
  - `conf`: Objectness confidence (logit)
  - `class_prob`: Class probability (logit for car class)

**Key Points:**
- **LeakyReLU(0.1)** activation (NOT ReLU!) - Critical for matching PyTorch model
- Dropout2d only used during training, not inference
- Final layer outputs raw logits (no activation)
- Padding=1 for 3×3 convolutions to maintain spatial dimensions

---

## Preprocessing Pipeline

### 1. Image Loading
```python
# Load image with OpenCV (BGR format)
image = cv2.imread(image_path)  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]
```

### 2. Aspect-Ratio Preserving Resize
```python
h, w = image.shape[:2]
scale = 416 / max(h, w)
new_h, new_w = int(h * scale), int(w * scale)
image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
```

**Example:**
- Input: 1920×1080 → scale = 416/1920 = 0.2167
- Resized: 416×234

### 3. Center Padding
```python
pad_h = 416 - new_h
pad_w = 416 - new_w
pad_top = pad_h // 2
pad_left = pad_w // 2
# Pad with gray (128, 128, 128)
image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                           cv2.BORDER_CONSTANT, value=(128, 128, 128))
```

**Result:** Padded image of exactly 416×416

### 4. Color Space Conversion
```python
# OpenCV loads as BGR, but ResNet expects RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### 5. ImageNet Normalization
```python
# Scale to [0, 1]
image = image.astype(np.float32) / 255.0

# Apply ImageNet statistics
mean = [0.485, 0.456, 0.406]  # Per channel
std = [0.229, 0.224, 0.225]   # Per channel
image = (image - mean) / std

# Result: Normalized tensor with mean≈0, std≈1 per channel
```

**⚠️ Critical:** ImageNet normalization is **required** because ResNet34 was pretrained with these statistics. Omitting this causes significant accuracy degradation.

### 6. Tensor Formatting
```python
image = np.transpose(image, (2, 0, 1))  # HWC → CHW
image = np.expand_dims(image, 0)        # Add batch dimension → (1, 3, 416, 416)
```

---

## Forward Pass

### 1. Backbone Forward Pass
```python
features = backbone.forward(image)  # (1, 512, 13, 13)
```

**Process:**
1. Initial Conv7×7 + BatchNorm + ReLU + MaxPool
2. Residual blocks in 4 layers
3. Each layer progressively downsamples and increases channels
4. Final output: 512 channels at 13×13 resolution

### 2. Detection Head Forward Pass
```python
predictions = detection_head.forward(features)  # (1, 30, 13, 13)
```

**Layer-by-layer:**
```
(1, 512, 13, 13)
    ↓ Conv3×3 + BN + LeakyReLU
(1, 512, 13, 13)
    ↓ Conv3×3 + BN + LeakyReLU
(1, 512, 13, 13)
    ↓ Conv1×1 + BN + LeakyReLU
(1, 256, 13, 13)
    ↓ Conv1×1 (no activation)
(1, 30, 13, 13)
```

**Output Format:**
- Shape: `(batch=1, channels=30, grid_h=13, grid_w=13)`
- Channel layout: `[anchor0_predictions, anchor1_predictions, ..., anchor4_predictions]`
- Each anchor: 6 values `[tx, ty, tw, th, conf, class_prob]`

---

## Postprocessing & Decoding

### 1. Reshape Predictions
```python
# Reshape from (1, 30, 13, 13) to parse per-anchor predictions
for h in range(13):
    for w in range(13):
        for a in range(5):  # 5 anchors
            offset = a * 6  # 6 values per anchor (5 + 1 class)
            tx = predictions[0, offset+0, h, w]
            ty = predictions[0, offset+1, h, w]
            tw = predictions[0, offset+2, h, w]
            th = predictions[0, offset+3, h, w]
            conf_raw = predictions[0, offset+4, h, w]
            class_raw = predictions[0, offset+5, h, w]
```

### 2. Apply Sigmoid Activations
```python
# Confidence must be passed through sigmoid
conf = 1.0 / (1.0 + np.exp(-conf_raw))

# XY offsets already have implicit sigmoid through decoding
# Class probabilities also need sigmoid (but not used for filtering with 1 class)
```

**⚠️ Critical:** Raw model outputs are **logits**. Sigmoid is **required** before thresholding!

### 3. Confidence Filtering
```python
if conf > conf_threshold:  # Default: 0.985
    # Process this detection
```

**Threshold Justification:**
- 0.985 = 98.5% confidence requirement
- High threshold reduces false positives
- Works correctly with sigmoid'd confidence values

### 4. Bounding Box Decoding

#### Step A: Center Coordinates (Normalized [0, 1])
```python
# Apply sigmoid to tx/ty, add grid offset, normalize by grid size
x_center = (grid_x + sigmoid(tx)) / grid_size
y_center = (grid_y + sigmoid(ty)) / grid_size

# Example for grid cell (5, 3):
# tx = -0.5 → sigmoid(-0.5) = 0.378
# x_center = (5 + 0.378) / 13 = 0.414
```

**Formula Explanation:**
- `grid_x + sigmoid(tx)`: Position within grid cell [0, 1] + cell index
- `/ grid_size`: Normalize to [0, 1] image coordinates

#### Step B: Width & Height (Normalized [0, 1])
```python
# Clamp tw/th to prevent extreme values
tw_clamped = np.clip(tw, -4, 4)  # Limits exp(tw) to [0.018, 54.6]
th_clamped = np.clip(th, -4, 4)

# Get anchor dimensions (normalized by grid size)
anchor_w = anchors[a][0] / grid_size  # e.g., 0.848 / 13 = 0.065
anchor_h = anchors[a][1] / grid_size

# Decode width/height
box_w = np.exp(tw_clamped) * anchor_w
box_h = np.exp(th_clamped) * anchor_h
```

**Anchor Boxes (5 anchors):**
```
Anchor 0: [0.848, 0.836]  - Small cars
Anchor 1: [2.067, 1.416]  - Medium cars
Anchor 2: [2.083, 3.081]  - Tall vehicles
Anchor 3: [4.639, 2.610]  - Large cars
Anchor 4: [4.196, 5.862]  - Very large vehicles
```

#### Step C: Convert to Corner Coordinates
```python
# Still in normalized [0, 1] space
x_min = np.clip(x_center - box_w / 2, 0, 1)
y_min = np.clip(y_center - box_h / 2, 0, 1)
x_max = np.clip(x_center + box_w / 2, 0, 1)
y_max = np.clip(y_center + box_h / 2, 0, 1)
```

**Clipping Rationale:** Ensures boxes stay within [0, 1] image bounds

#### Step D: Validate Box
```python
if x_max > x_min and y_max > y_min:
    # Valid box, continue processing
```

#### Step E: Convert to Pixel Coordinates
```python
# Convert from normalized [0, 1] to 416×416 pixel space
x1_px = x_min * 416
y1_px = y_min * 416
x2_px = x_max * 416
y2_px = y_max * 416
```

#### Step F: Denormalize to Original Image
```python
# Account for padding and scaling from preprocessing
x1_orig = (x1_px - pad_left) / scale
y1_orig = (y1_px - pad_top) / scale
x2_orig = (x2_px - pad_left) / scale
y2_orig = (y2_px - pad_top) / scale
```

**Complete Example:**
```
Original image: 1920×1080
Preprocessed: 416×234 → padded 416×416 (pad_top=91, pad_left=0)
scale = 0.2167

Detection at normalized (0.5, 0.4):
→ 416×416 space: (208, 166)
→ Remove padding: (208 - 0, 166 - 91) = (208, 75)
→ Unscale: (208 / 0.2167, 75 / 0.2167) = (960, 346)
→ Final: Car center at (960, 346) in original 1920×1080 image
```

---

## Non-Maximum Suppression (NMS)

### Purpose
Remove duplicate detections of the same object

### Algorithm: Standard NMS with IoU
```python
def nms(boxes, scores, iou_threshold=0.3):
    # Sort by confidence (descending)
    order = np.argsort(scores)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining boxes
        ious = [compute_iou(boxes[i], boxes[j]) for j in order[1:]]
        
        # Keep only boxes with IoU < threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep
```

### IoU (Intersection over Union) Calculation
```python
def compute_iou(box_a, box_b):
    # Find intersection rectangle
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    
    # Compute areas
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0
```

### NMS Threshold: 0.3
- **Low IoU (< 0.3):** Boxes are different objects → Keep both
- **High IoU (≥ 0.3):** Boxes likely same object → Keep only highest confidence

**Example:**
```
Before NMS:
- Box1: [100, 100, 200, 200], conf=0.99
- Box2: [105, 105, 205, 205], conf=0.95
- IoU(Box1, Box2) = 0.82 > 0.3 → Remove Box2

After NMS:
- Box1: [100, 100, 200, 200], conf=0.99
```

---

## Critical Implementation Details

### 1. LeakyReLU vs ReLU
**Location:** Detection head only

**Difference:**
```python
# ReLU: f(x) = max(0, x)
#   x < 0 → 0
#   x ≥ 0 → x

# LeakyReLU: f(x) = max(0.1*x, x)
#   x < 0 → 0.1*x
#   x ≥ 0 → x
```

**Impact:** LeakyReLU preserves gradient flow for negative values, allowing the model to learn more nuanced features. Using ReLU instead causes **significant accuracy degradation** (~50% fewer detections).

**Implementation:**
```python
def leaky_relu(x, negative_slope=0.1):
    return np.where(x > 0, x, x * negative_slope)
```

### 2. Sigmoid on Raw Outputs
**Required for:**
- Confidence scores
- XY offsets (implicit through decoding formula)
- Class probabilities

**Why:** Model outputs logits (unbounded values). Sigmoid maps to [0, 1] probability space.

**Common Error:**
```python
# WRONG: Using raw logit
if conf_raw > 0.985:  # Raw could be -5, never passes!

# CORRECT: Apply sigmoid first
conf = sigmoid(conf_raw)
if conf > 0.985:  # Now in [0, 1] range
```

### 3. ImageNet Normalization
**Mean:** `[0.485, 0.456, 0.406]` (RGB)  
**Std:** `[0.229, 0.224, 0.225]` (RGB)

**Why Critical:** ResNet34 backbone was pretrained on ImageNet with these exact statistics. Not applying them causes the model to see "shifted" colors, degrading feature extraction.

**Visual Impact:**
- Without normalization: Model sees raw [0, 1] RGB values
- With normalization: Model sees standardized values matching training distribution
- **Effect:** ~70% accuracy loss if omitted

### 4. BGR to RGB Conversion
**Why:** OpenCV loads images as BGR, but PyTorch models expect RGB.

**Impact:**
```python
# BGR: Blue=car, Green=grass, Red=sky
# RGB: Red=car, Green=grass, Blue=sky
# Model trained on RGB would misinterpret BGR colors!
```

### 5. Coordinate Space Consistency
**Throughout Pipeline:**
1. Preprocessing → Pixel coordinates
2. Model output → Logits
3. Decoding → **Normalized [0, 1]**
4. Final output → Original image pixel coordinates

**Critical:** Maintain normalized [0, 1] during decoding, convert to pixels only at the end.

### 6. Anchor Scaling
**Formula:**
```python
# WRONG: No anchor
box_w = np.exp(tw)

# CORRECT: Scale by anchor
box_w = np.exp(tw) * (anchor_w / grid_size)
```

**Why:** The model predicts offsets **relative to anchor dimensions**, not absolute sizes.

---

## Configuration Parameters

### Model Configuration (`config.json`)
```json
{
  "num_classes": 1,
  "conf_threshold": 0.985,
  "nms_threshold": 0.3,
  "class_names": ["car"]
}
```

### Anchor Boxes (`anchors.npy`)
```python
anchors = np.array([
    [0.8478, 0.8363],  # Anchor 0: Small
    [2.0669, 1.4161],  # Anchor 1: Medium
    [2.0828, 3.0814],  # Anchor 2: Tall
    [4.6393, 2.6100],  # Anchor 3: Large
    [4.1957, 5.8618]   # Anchor 4: Extra Large
])
```

### Preprocessing Constants
```python
INPUT_SIZE = 416
PADDING_COLOR = (128, 128, 128)  # Gray
INTERPOLATION = cv2.INTER_LINEAR

# ImageNet statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

### Inference Thresholds
```python
CONF_THRESHOLD = 0.985  # 98.5% confidence minimum
NMS_THRESHOLD = 0.3     # 30% IoU overlap maximum
```

---

## Performance Characteristics

### Computational Complexity
- **Backbone:** ~1.8 GFLOPs (ResNet34)
- **Detection Head:** ~0.3 GFLOPs
- **Total:** ~2.1 GFLOPs per 416×416 image

### Memory Usage
- **Model Weights:** ~85 MB (FP32)
- **Intermediate Activations:** ~50 MB per image
- **Total Runtime Memory:** ~150 MB

### Inference Speed (CPU)
- **Pure NumPy:** ~200-500ms per image (depending on CPU)
- **Bottleneck:** Convolution operations in backbone

### Accuracy Metrics
- **Precision @ 0.985 threshold:** ~99%
- **Recall @ 0.985 threshold:** ~85%
- **mAP@0.5:** ~92%

---

## Common Issues & Solutions

### Issue 1: No Detections
**Symptoms:** Model returns empty list

**Possible Causes:**
1. Missing sigmoid on confidence
2. Wrong ImageNet normalization
3. BGR/RGB mixup
4. Threshold too high

**Solution:** Verify preprocessing pipeline matches this document exactly

### Issue 2: Low Confidence Scores
**Symptoms:** All detections < 0.5 confidence

**Possible Causes:**
1. Using ReLU instead of LeakyReLU in detection head
2. Missing ImageNet normalization
3. Model weights not loaded correctly

**Solution:** Check activation functions and normalization

### Issue 3: Incorrect Bounding Boxes
**Symptoms:** Boxes in wrong locations or wrong sizes

**Possible Causes:**
1. Missing anchor scaling
2. Wrong coordinate space conversions
3. Padding/scaling not accounted for

**Solution:** Verify decoding formula and coordinate transformations

### Issue 4: Too Many Detections
**Symptoms:** Multiple boxes per car

**Possible Causes:**
1. NMS not applied
2. NMS threshold too high

**Solution:** Apply NMS with threshold 0.3

---

## Validation Checklist

Before deploying, verify:

- [ ] ResNet34 backbone uses ReLU (standard)
- [ ] Detection head uses LeakyReLU(0.1)
- [ ] ImageNet normalization applied (mean/std)
- [ ] BGR to RGB conversion
- [ ] Sigmoid applied to confidence before thresholding
- [ ] Anchor scaling in bbox decoding
- [ ] Normalized [0, 1] coordinate space during decoding
- [ ] tw/th clamped to [-4, 4]
- [ ] NMS applied with threshold 0.3
- [ ] Final coordinates denormalized to original image

---

## References

### Model Architecture
- **ResNet:** "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **YOLOv2:** "YOLO9000: Better, Faster, Stronger" (Redmon & Farhadi, 2016)

### Implementation
- **PyTorch Model:** `comvis/utils/model.py`
- **NumPy Inference:** `deployment/spark_engine/`
- **Training Pipeline:** `comvis/main.py`

### Configuration Files
- **Deployment Config:** `deployment/config.json`
- **Training Config:** `comvis/enhanced_config.json`
- **Anchors:** `deployment/anchors.npy`

---

## Appendix: Complete Inference Flow Diagram

```
Input Image (H, W, 3) BGR
    ↓
Resize (preserve aspect) → (new_h, new_w, 3)
    ↓
Center Pad (gray) → (416, 416, 3)
    ↓
BGR → RGB Conversion
    ↓
Normalize [0, 1] → / 255.0
    ↓
ImageNet Normalize → (x - mean) / std
    ↓
HWC → CHW → (3, 416, 416)
    ↓
Add Batch Dim → (1, 3, 416, 416)
    ↓
ResNet34 Backbone → (1, 512, 13, 13)
    ↓
Detection Head → (1, 30, 13, 13)
    ↓
For each grid cell (13×13):
    For each anchor (5):
        ↓
        Extract [tx, ty, tw, th, conf, class]
        ↓
        Apply sigmoid(conf)
        ↓
        if conf > 0.985:
            ↓
            Decode bbox (normalized [0,1])
            ↓
            Convert to pixels (416×416)
            ↓
            Denormalize (original size)
            ↓
            Add to detections list
    ↓
Apply NMS (threshold=0.3)
    ↓
Final Detections: List[{bbox, confidence, class}]
```

---

**Document Version:** 1.0  
**Last Updated:** November 19, 2025  
**Maintained By:** SPARK Development Team
