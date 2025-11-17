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
from roboflow import Roboflow
print("Imports successful. Torch version:", torch.__version__, "FiftyOne version:", fo.__version__)

local_coco_path = os.path.join(os.getcwd(), "datasets", "COCO_car")

#%% Configuration
coco_max_samples = 12000
enable_coco = True  # Set to False to skip COCO dataset
enable_roboflow = True  # Set to True to include Roboflow datasets
enable_filtering = True  # Set to False to use original dataset without filtering

# Roboflow Configuration (if enable_roboflow = True)
ROBOFLOW_API_KEY = "ECe6lJGRmHj38vmxj17r"  # Replace with your single Roboflow API key

# Add your Roboflow datasets here (they will all use the same API key above)
roboflow_datasets = [
    {
        "workspace": "aiml-oydm4",
        "project": "car-ua0ro",
        "version": 4,
        "name": "aiml_cars",
        "use_all_classes": True  # Use all classes from this dataset
    },
    {
        "workspace": "plane-uurb7",
        "project": "car-5oqis",
        "version": 1,
        "name": "plane_cars",
        "use_all_classes": True  # Use all classes from this dataset
    },
    {
        "workspace": "hng",
        "project": "car-uvjtt",
        "version": 5,
        "name": "hng_cars",
        "use_all_classes": False  # Only use 'car' class from this dataset
    },
]
#%% Load COCO Dataset
coco_dataset = None
if enable_coco:
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
        print(f"Failed to load COCO dataset: {str(e)}")
        coco_dataset = None
        enable_coco = False

#%% Load Roboflow Datasets
def load_roboflow_dataset(api_key, workspace, project, version, name):
    """Load a dataset from Roboflow"""
    try:
        print(f"Loading Roboflow dataset: {name}...")
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("coco")
        
        # Get the download path
        download_path = os.path.join(os.getcwd(), project + "-" + str(version))
        print(f"Downloaded Roboflow dataset '{name}' to {download_path}")
        
        return download_path, name
    except Exception as e:
        print(f"Failed to load Roboflow dataset '{name}': {str(e)}")
        return None, None

def convert_roboflow_to_fiftyone(roboflow_path, dataset_name, use_all_classes=True):
    """Convert Roboflow COCO format to FiftyOne dataset"""
    try:
        # Roboflow downloads datasets with train/valid/test splits
        train_path = os.path.join(roboflow_path, "train")
        
        if not os.path.exists(train_path):
            print(f"Train directory not found: {train_path}")
            return None
        
        # Load annotations
        ann_file = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.exists(ann_file):
            print(f"Annotations file not found: {ann_file}")
            return None
        
        # Create FiftyOne dataset from COCO format
        fo_dataset = fo.Dataset.from_dir(
            dataset_dir=train_path,
            dataset_type=fo.types.COCODetectionDataset,
            label_field="detections",
            name=f"roboflow_{dataset_name}"
        )
        
        if use_all_classes:
            # Keep all classes but normalize them to "car"
            print(f"Using all classes from '{dataset_name}' and normalizing to 'car'")
            def normalize_all_to_car(sample):
                if sample.detections:
                    for detection in sample.detections.detections:
                        detection.label = "car"
                return sample
            
            fo_dataset = fo_dataset.map(normalize_all_to_car)
            fo_dataset = fo_dataset.match(fo.ViewField("detections.detections").length() > 0)
        else:
            # Only keep car-related classes
            print(f"Filtering only car-related classes from '{dataset_name}'")
            car_classes = ["car", "vehicle", "automobile", "Car", "Vehicle"]
            
            # Normalize labels to "car"
            def normalize_labels(sample):
                if sample.detections:
                    for detection in sample.detections.detections:
                        if detection.label.lower() in [c.lower() for c in car_classes]:
                            detection.label = "car"
                return sample
            
            fo_dataset = fo_dataset.map(normalize_labels)
            
            # Filter to keep only car detections
            fo_dataset = fo_dataset.filter_labels("detections", fo.ViewField("label") == "car")
            fo_dataset = fo_dataset.match(fo.ViewField("detections.detections").length() > 0)
        
        print(f"Loaded Roboflow dataset '{dataset_name}' with {len(fo_dataset)} samples")
        return fo_dataset
        
    except Exception as e:
        print(f"Failed to convert Roboflow dataset '{dataset_name}': {str(e)}")
        return None

roboflow_fo_datasets = []
if enable_roboflow:
    for rb_config in roboflow_datasets:
        download_path, name = load_roboflow_dataset(
            ROBOFLOW_API_KEY,  # Use the single API key defined in configuration
            rb_config["workspace"],
            rb_config["project"],
            rb_config["version"],
            rb_config["name"]
        )
        
        if download_path:
            use_all_classes = rb_config.get("use_all_classes", True)  # Default to True
            fo_dataset = convert_roboflow_to_fiftyone(download_path, name, use_all_classes)
            if fo_dataset:
                # Add source field to each sample
                fo_dataset.set_values("source", [f"roboflow_{name}"] * len(fo_dataset))
                roboflow_fo_datasets.append(fo_dataset)
    
    print(f"Successfully loaded {len(roboflow_fo_datasets)} Roboflow datasets")

#%% Combine Datasets
datasets_to_combine = []
dataset_sources = []

# Add COCO if enabled
if enable_coco and coco_dataset is not None:
    # Clone the dataset and add source field
    coco_dataset.clone()  # Ensure we have a persistent dataset
    coco_dataset.set_values("source", ["coco"] * len(coco_dataset))
    datasets_to_combine.append(coco_dataset)
    dataset_sources.append(f"COCO ({len(coco_dataset)} samples)")

# Add Roboflow datasets if any
if enable_roboflow and roboflow_fo_datasets:
    for fo_ds in roboflow_fo_datasets:
        # Get source name from the dataset
        if len(fo_ds) > 0:
            # The source was already set when we created the dataset
            source_name = fo_ds.name
        else:
            source_name = "roboflow"
        datasets_to_combine.append(fo_ds)
        dataset_sources.append(f"{source_name} ({len(fo_ds)} samples)")

# Combine all datasets
if len(datasets_to_combine) == 0:
    raise ValueError("No datasets loaded! Enable at least one dataset source.")
elif len(datasets_to_combine) == 1:
    print(f"Using single dataset: {dataset_sources[0]}")
    export_dataset = datasets_to_combine[0]
    dataset_name_suffix = "COCO" if enable_coco else "Roboflow"
else:
    print(f"Combining {len(datasets_to_combine)} datasets...")
    for source in dataset_sources:
        print(f"  - {source}")
    
    # Combine all datasets
    export_dataset = datasets_to_combine[0]
    for ds in datasets_to_combine[1:]:
        export_dataset = export_dataset.concat(ds)
    
    print(f"Combined dataset: {len(export_dataset)} total samples")
    
    dataset_name_suffix = "Combined"

# Set export path
local_export_path = os.path.join(os.getcwd(), "datasets", f"temp_{dataset_name_suffix}")

#%% Export combined dataset locally
export_dataset.export(
    export_dir=local_export_path,
    dataset_type=fo.types.COCODetectionDataset,
    overwrite=True
)
print(f"Combined dataset exported to {local_export_path}")

#%% Dataset Filtering Functions
def analyze_car_annotation(ann, img_width, img_height, min_size=16, max_size=300):
    """Analyze car annotation quality"""
    x_min, y_min, width, height = ann['bbox']
    
    # Convert to absolute pixels
    abs_width = width
    abs_height = height
    area = abs_width * abs_height
    aspect_ratio = abs_width / abs_height if abs_height > 0 else 0
    
    # Quality checks
    is_tiny = abs_width < min_size or abs_height < min_size
    is_huge = abs_width > max_size or abs_height > max_size
    bad_aspect = aspect_ratio < 0.3 or aspect_ratio > 4.0
    too_small_area = area < (min_size * min_size)
    invalid_bbox = (width <= 0 or height <= 0 or 
                   x_min < 0 or y_min < 0 or 
                   x_min + width > img_width or y_min + height > img_height)
    
    return {
        'is_valid': not (is_tiny or is_huge or bad_aspect or too_small_area or invalid_bbox),
        'abs_width': abs_width,
        'abs_height': abs_height,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'is_tiny': is_tiny,
        'is_huge': is_huge,
        'bad_aspect': bad_aspect
    }

def filter_dataset_samples(car_samples, min_car_size=16, max_car_size=300):
    """Filter dataset samples to remove problematic annotations"""
    print(f"\n{'='*60}")
    print("FILTERING DATASET")
    print(f"{'='*60}")
    
    filtered_samples = []
    total_cars = 0
    valid_cars = 0
    tiny_cars = 0
    huge_cars = 0
    bad_aspect_cars = 0
    
    for i, (img, annotations) in enumerate(car_samples):
        img_width, img_height = img.size
        
        # Skip images that are too small
        if img_width < 64 or img_height < 64:
            continue
        
        valid_annotations = []
        
        for ann in annotations:
            total_cars += 1
            
            analysis = analyze_car_annotation(ann, img_width, img_height, min_car_size, max_car_size)
            
            if analysis['is_valid']:
                valid_annotations.append(ann)
                valid_cars += 1
            else:
                # Count reasons for filtering
                if analysis['is_tiny']:
                    tiny_cars += 1
                if analysis['is_huge']:
                    huge_cars += 1
                if analysis['bad_aspect']:
                    bad_aspect_cars += 1
        
        # Keep samples with at least one valid car
        if valid_annotations:
            filtered_samples.append((img, valid_annotations))
    
    # Print filtering statistics
    print(f"Original samples: {len(car_samples)}")
    print(f"Filtered samples: {len(filtered_samples)}")
    print(f"Original cars: {total_cars}")
    print(f"Valid cars: {valid_cars}")
    print(f"Removed cars: {total_cars - valid_cars} ({(total_cars - valid_cars)/total_cars*100:.1f}%)")
    
    print(f"\nFiltered out:")
    print(f"- Tiny cars (< {min_car_size}px): {tiny_cars}")
    print(f"- Huge cars (> {max_car_size}px): {huge_cars}")
    print(f"- Bad aspect ratios: {bad_aspect_cars}")
    
    return filtered_samples
#%% Load and Process Combined Dataset
root_dir = os.path.join(local_export_path, "data")
ann_file = os.path.join(local_export_path, "labels.json")

if not os.path.exists(root_dir) or not os.path.exists(ann_file):
    raise ValueError(f"Combined data not found at {root_dir} or {ann_file} after export.")

dataset = CocoDetection(root_dir, ann_file, transform=None)
print(f"Loaded combined dataset with {len(dataset)} total samples")

car_samples = []
for i in range(len(dataset)):
    car_samples.append(dataset[i])
print(f"Loaded combined dataset with {len(car_samples)} samples")

#%% Apply Filtering (Optional)
if enable_filtering:
    print("Applying filtering to dataset...")
    filtered_car_samples = filter_dataset_samples(car_samples, min_car_size=16, max_car_size=300)
    # Use filtered samples for the rest of the pipeline
    car_samples = filtered_car_samples
    
    # Determine dataset name based on sources
    if len(datasets_to_combine) > 1:
        dataset_name = "Enhanced_Combined"
    elif enable_roboflow:
        dataset_name = "Enhanced_Roboflow"
    else:
        dataset_name = "Enhanced_COCO"
else:
    print("Using unfiltered dataset...")
    
    # Determine dataset name based on sources
    if len(datasets_to_combine) > 1:
        dataset_name = "Combined"
    elif enable_roboflow:
        dataset_name = "Roboflow"
    else:
        dataset_name = "COCO_car"

#%% Split and Export Enhanced Dataset
total_samples = len(car_samples)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
print(f"Splitting {dataset_name} dataset: train={train_size}, val={val_size}, test={test_size}")

# Shuffle dataset
indices = np.random.permutation(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Export to YOLO format with enhanced validation
output_dir = os.path.join(os.getcwd(), "datasets", dataset_name, "parsed_dataset")

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
# Enhanced anchor computation
def compute_enhanced_anchors(car_samples, img_size=416, grid_size=13):
    """Compute anchors from enhanced dataset with better size distribution"""
    print(f"\n{'='*60}")
    print("COMPUTING ANCHORS FROM ENHANCED DATASET")
    print(f"{'='*60}")
    
    bbox_sizes = []
    size_stats = []
    
    for img, tgt in car_samples:
        img_width, img_height = img.size
        
        for ann in tgt:
            # Get bbox in COCO format [x, y, width, height]
            x_min, y_min, width, height = ann['bbox']
            
            # Validate bbox (should already be valid from filtering)
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
            size_stats.append({
                'abs_width': width,
                'abs_height': height,
                'grid_width': grid_width,
                'grid_height': grid_height
            })
    
    if not bbox_sizes:
        # Fallback anchors if no valid boxes found
        print("No valid boxes found, using fallback anchors")
        return np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                        [9.47112, 4.84053], [11.2364, 10.0071]])
    
    bbox_sizes = np.array(bbox_sizes)
    print(f"Extracted {len(bbox_sizes)} bounding box sizes for k-means")
    
    # Print enhanced statistics
    abs_widths = [s['abs_width'] for s in size_stats]
    abs_heights = [s['abs_height'] for s in size_stats]
    grid_widths = [s['grid_width'] for s in size_stats]
    grid_heights = [s['grid_height'] for s in size_stats]
    
    print(f"Absolute size range - Width: {min(abs_widths):.1f}-{max(abs_widths):.1f}px, Height: {min(abs_heights):.1f}-{max(abs_heights):.1f}px")
    print(f"Grid size range - Width: {min(grid_widths):.2f}-{max(grid_widths):.2f}, Height: {min(grid_heights):.2f}-{max(grid_heights):.2f}")
    print(f"Average grid size - Width: {np.mean(grid_widths):.2f}, Height: {np.mean(grid_heights):.2f}")
    
    # Use k-means to find 5 anchor boxes
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(bbox_sizes)
    anchors = kmeans.cluster_centers_
    
    # Sort anchors by area
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    print(f"Enhanced anchors (grid units): {anchors}")
    
    return anchors

# Compute enhanced anchors
enhanced_anchors = compute_enhanced_anchors(car_samples, img_size=416, grid_size=13)

# Save enhanced anchors
anchor_file = os.path.join(os.getcwd(), "datasets", dataset_name, "anchors.npy")
os.makedirs(os.path.dirname(anchor_file), exist_ok=True)
np.save(anchor_file, enhanced_anchors)
print(f"Enhanced anchors saved to {anchor_file}")

#%% Final Verification
print(f"\n{'='*60}")
print("ENHANCED DATASET VERIFICATION")
print(f"{'='*60}")

# Load and display anchors
anchors = np.load(anchor_file)
print("Enhanced Anchors:", anchors)

# Verify dataset structure
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

dataset_description = f"{dataset_name} dataset"
if enable_filtering:
    dataset_description += " (with filtering)"

source_descriptions = []
if enable_coco:
    source_descriptions.append("COCO")
if enable_roboflow and roboflow_fo_datasets:
    source_descriptions.append(f"{len(roboflow_fo_datasets)} Roboflow dataset(s)")

if len(source_descriptions) > 1:
    dataset_description += f" ({' + '.join(source_descriptions)})"

print(f"\n{dataset_description} preparation complete! Key improvements:")
print("✅ Enhanced dataset with better quality control")
if enable_filtering:
    print("✅ Filtered out tiny and problematic car annotations")
print("✅ Enhanced anchors computed from quality data")
if len(datasets_to_combine) > 1:
    print(f"✅ Combined {len(datasets_to_combine)} datasets for more diverse training data")
if enable_roboflow:
    print("✅ Integrated custom Roboflow datasets")
print("✅ Optimized for automotive detection scenarios")
print("✅ Ready for training with reduced false positives!")
# %%
