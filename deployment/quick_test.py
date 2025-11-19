import sys
import os
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from spark_engine.core import SPARKEngine

# Initialize engine
print("Initializing SPARK engine...")
engine = SPARKEngine(
    model_path=r"e:\projects\SPARK\comvis\models\best_model.npz",
    config_path="config.json",
    anchor_path="anchors.npy"
)

# Threshold is now set in config.json
print(f"Using threshold: {engine.conf_threshold}")

# Test with an image
test_image = r"e:\projects\SPARK\comvis\test_images\temp.jpg"
if not os.path.exists(test_image):
    print(f"Test image not found: {test_image}")
    sys.exit(1)

print(f"\nTesting with: {test_image}")
image = np.array(Image.open(test_image).convert("RGB"))

# Run inference
detections = engine.predict(image)

print(f"\nTotal detections: {len(detections)}")
for i, det in enumerate(detections[:5]):  # Show first 5
    print(f"  {i+1}. {det['class_name']}: {det['confidence']:.4f} at {det['bbox']}")
