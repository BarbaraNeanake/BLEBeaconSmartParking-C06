# SPARK Car Detection - Quick Reference

## Setup & Usage

### 1. Train Model
```python
# Run in main.py
device = setup_device()
config = load_config()
train_pipeline(config, device)
```

### 2. Test Model
```python
# Run in main.py
inference_pipeline(config, device)
```

## Key Settings

### Configuration (`utils/config.py`)
```python
img_size: 416              # Image size
num_epochs: 50             # Training epochs
batch_size: 8              # Batch size
learning_rate: 0.001       # Learning rate
conf_threshold: 0.5        # Detection confidence
nms_threshold: 0.45        # NMS threshold
```

### Adjust Detection Sensitivity
- **Too many false detections?** → Increase `conf_threshold` to 0.6-0.7
- **Missing real cars?** → Decrease `conf_threshold` to 0.4

## Fixed Issues
✓ No more empty space detections (increased conf threshold)  
✓ Boxes stay reasonable size (added exp clipping)  
✓ Faster training (optimized settings)  
✓ Simplified code (removed verbose prints)  

## File Structure
```
main.py              # Training & testing scripts
utils/
  config.py          # Configuration
  model.py           # YOLOv2-ResNet50
  train_utils.py     # Training logic
  inference_utils.py # Testing logic
  data_utils.py      # Data loading
  loss.py            # Loss functions
```
