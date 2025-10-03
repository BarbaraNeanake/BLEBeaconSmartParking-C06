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

#%% preview with FiftyOne
fo.launch_app(coco_dataset, auto=False)

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
print(f"Loaded COCO dataset with {sum(1 for _ in car_samples)} samples")  # Count without storing

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

# Export to YOLO format
output_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "parsed_dataset")
os.makedirs(output_dir, exist_ok=True)
for split, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    for i in indices:
        img, tgt = car_samples[i]
        img_name = f"{i}.jpg"
        img_path = os.path.join(split_dir, "images", img_name)
        img.save(img_path)  # Directly save PIL Image
        label_path = os.path.join(split_dir, "labels", f"{i}.txt")
        with open(label_path, 'w') as f:
            for ann in tgt:
                x_min, y_min, width, height = ann['bbox']
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                img_width, img_height = 640, 480  # Adjust if needed
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
    print(f"Exported {split} split with {len(indices)} samples to {split_dir}")
# %%
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
print(f"Computed anchors: {anchors}")

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