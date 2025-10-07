#%% [markdown]
# Cell 1: Imports

import os
import json
import numpy as np
from PIL import Image
import torch
import fiftyone as fo
import fiftyone.zoo as foz
from torchvision.datasets import CocoDetection
from sklearn.cluster import KMeans
print("Imports successful. Torch version:", torch.__version__, "FiftyOne version:", fo.__version__)

local_coco_path = os.path.join(os.getcwd(), "datasets", "COCO_car")
#%%
coco_max_samples = 12000
try:
    coco_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        classes=["car"],
        label_field="detections",
        # max_samples=coco_max_samples
    ).filter_labels("detections", fo.ViewField("label") == "car").limit(coco_max_samples)
    print(f"Loaded COCO dataset with {len(coco_dataset)} car samples")
except Exception as e:
    raise ValueError(f"Failed to load COCO dataset: {str(e)}")

#%% Export filtered dataset locally
coco_dataset.export(
    export_dir=local_coco_path,
    dataset_type=fo.types.COCODetectionDataset,
    overwrite=True
)
print(f"Filtered COCO dataset exported to {local_coco_path}")
#%%
root_dir = os.path.join(local_coco_path, "data")
ann_file = os.path.join(local_coco_path, "labels.json")
if not os.path.exists(root_dir) or not os.path.exists(ann_file):
    raise ValueError(f"COCO data not found at {root_dir} or {ann_file} after export.")

dataset = CocoDetection(root_dir, ann_file, transform=None)
print(f"Loaded COCO dataset with {len(dataset)} total samples")

car_samples = list(dataset)
print(f"Loaded COCO dataset with {sum(1 for _ in car_samples)} samples")

#%%
total_samples = len(car_samples)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
print(f"Splitting: train={train_size}, val={val_size}, test={test_size}")

# Shuffle dataset
indices = np.random.permutation(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Export to YOLO format with improved validation
output_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "parsed_dataset")
os.makedirs(output_dir, exist_ok=True)
for split, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    
    valid_samples = 0
    for i in indices:
        img, tgt = car_samples[i]
        img_width, img_height = img.size
        
        # Skip images that are too small
        if img_width < 64 or img_height < 64:
            continue
        
        # Filter valid annotations
        valid_annotations = []
        for ann in tgt:
            x_min, y_min, width, height = ann['bbox']
            
            # Validate bbox
            if width <= 0 or height <= 0:
                continue
            if x_min < 0 or y_min < 0 or x_min + width > img_width or y_min + height > img_height:
                continue
            
            # Calculate normalized coordinates
            x_center = (x_min + width / 2) / img_width
            y_center = (y_min + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            # Ensure coordinates are in valid range
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= norm_width <= 1 and 0 <= norm_height <= 1):
                continue
            
            valid_annotations.append([x_center, y_center, norm_width, norm_height])
        
        # Skip images with no valid annotations
        if not valid_annotations:
            continue
        
        # Save image and labels
        img_name = f"{i:06d}.jpg"
        img_path = os.path.join(split_dir, "images", img_name)
        img.save(img_path, quality=95)
        
        label_path = os.path.join(split_dir, "labels", f"{i:06d}.txt")
        with open(label_path, 'w') as f:
            for x_center, y_center, norm_width, norm_height in valid_annotations:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        valid_samples += 1
    
    print(f"Exported {split} split with {valid_samples} valid samples to {split_dir}")
# %%
# Improved anchor computation
def compute_anchors_properly(car_samples, img_size=416, grid_size=13):
    """Compute anchors with proper scaling"""
    bbox_sizes = []
    
    for img, tgt in car_samples:
        img_width, img_height = img.size
        
        for ann in tgt:
            # Get bbox in COCO format [x, y, width, height]
            x_min, y_min, width, height = ann['bbox']
            
            # Validate bbox
            if width <= 0 or height <= 0:
                continue
            if x_min < 0 or y_min < 0 or x_min + width > img_width or y_min + height > img_height:
                continue
            
            # Normalize to image dimensions
            norm_width = width / img_width
            norm_height = height / img_height
            
            # Scale to grid size (this is what YOLO anchors represent)
            grid_width = norm_width * grid_size
            grid_height = norm_height * grid_size
            
            bbox_sizes.append([grid_width, grid_height])
    
    if not bbox_sizes:
        # Fallback anchors if no valid boxes found
        return np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                        [9.47112, 4.84053], [11.2364, 10.0071]])
    
    bbox_sizes = np.array(bbox_sizes)
    print(f"Extracted {len(bbox_sizes)} bounding box sizes for k-means")
    print(f"Bbox size statistics - Mean: {bbox_sizes.mean(axis=0)}, Std: {bbox_sizes.std(axis=0)}")
    
    # Use k-means to find 5 anchor boxes
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(bbox_sizes)
    anchors = kmeans.cluster_centers_
    
    # Sort anchors by area
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    return anchors

bbox_sizes = []
for _, tgt in car_samples:
    for ann in tgt:
        width, height = ann['bbox'][2], ann['bbox'][3]
        bbox_sizes.append([width * 13 / 640, height * 13 / 480])  # Scale to 13x13 grid
if not bbox_sizes:
    raise ValueError("No bounding boxes found for anchor computation")
bbox_sizes = np.array(bbox_sizes)
print(f"Extracted {len(bbox_sizes)} bounding box sizes for k-means")
kmeans = KMeans(n_clusters=5, random_state=0).fit(bbox_sizes)
anchors = kmeans.cluster_centers_
# Compute anchors with improved method
anchors = compute_anchors_properly(car_samples, img_size=416, grid_size=13)

# Save anchors for use in train.py
anchor_output_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "anchors.npy")
np.save(anchor_output_dir, anchors)
print(f"Anchors saved to {anchor_output_dir}")

#%%
anchors = np.load(anchor_output_dir)
print("Loaded Anchors:", anchors)

for split in ["train", "val", "test"]:
    img_dir = os.path.join(output_dir, split, "images")
    label_dir = os.path.join(output_dir, split, "labels")
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"WARNING: {split} split directories missing: {img_dir} or {label_dir}")
        continue
    img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    label_count = len([f for f in os.listdir(label_dir) if f.lower().endswith(".txt")])
    print(f"{split} split: {img_count} images, {label_count} labels")
    if img_count != label_count:
        print(f"WARNING: Mismatch in {split} split - {img_count} images vs {label_count} labels")

print("Dataset verification complete. Ready for training with pre-trained Darknet19 in train.py.")