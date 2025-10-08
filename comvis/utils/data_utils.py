"""
Data utilities for SPARK car detection pipeline
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import cv2
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional


class ParsedYOLODataset(Dataset):
    """
    Parsed YOLO dataset with enhanced data augmentation and validation
    """
    
    def __init__(self, root_dir: str, split: str, config, transform=None, augment: bool = True):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.label_dir = os.path.join(root_dir, split, "labels")
        self.transform = transform
        self.config = config
        self.img_size = config.img_size
        self.augment = augment and split == "train"
        
        # Load anchors
        anchors_path = os.path.join(config.dataset_root, "anchors.npy")
        if os.path.exists(anchors_path):
            self.anchors = np.load(anchors_path)
        else:
            # Fallback anchors
            self.anchors = np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                                   [9.47112, 4.84053], [11.2364, 10.0071]])
            print(f"Warning: Using fallback anchors for {split} split")
        
        self.num_anchors = len(self.anchors)
        self.grid_size = config.grid_size

        # Get image files
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.image_files.sort()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")
        
        print(f"Loaded {len(self.image_files)} images for {split} split")

    def find_best_anchor(self, box_wh: List[float]) -> int:
        """Find the best anchor for a given box using IoU"""
        box_wh = np.array(box_wh)
        anchor_wh = self.anchors
        
        # Calculate IoU between box and each anchor
        intersection = np.minimum(box_wh[0], anchor_wh[:, 0]) * np.minimum(box_wh[1], anchor_wh[:, 1])
        box_area = box_wh[0] * box_wh[1]
        anchor_areas = anchor_wh[:, 0] * anchor_wh[:, 1]
        union = box_area + anchor_areas - intersection
        
        # Avoid division by zero
        union = np.maximum(union, 1e-8)
        iou = intersection / union
        return np.argmax(iou)

    def augment_image_and_boxes(self, image: Image.Image, boxes: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Apply data augmentation"""
        if not self.augment or len(boxes) == 0:
            return image, boxes

        # Convert PIL to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img_array = cv2.flip(img_array, 1)
            # Flip x coordinates
            boxes[:, 1] = 1.0 - boxes[:, 1]

        # Random scaling and translation
        if random.random() > 0.7:
            scale_factor = random.uniform(0.9, 1.1)
            h, w = img_array.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # Resize image
            img_array = cv2.resize(img_array, (new_w, new_h))
            
            # Handle crop/pad
            if scale_factor > 1.0:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img_array = img_array[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                img_array = cv2.copyMakeBorder(img_array, pad_y, h-new_h-pad_y, 
                                             pad_x, w-new_w-pad_x, cv2.BORDER_CONSTANT)

        # Convert back to PIL
        image = Image.fromarray(img_array)

        # Color augmentations
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.9, 1.1)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

        if random.random() > 0.5:
            contrast_factor = random.uniform(0.9, 1.1)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

        if random.random() > 0.5:
            saturation_factor = random.uniform(0.9, 1.1)
            image = transforms.functional.adjust_saturation(image, saturation_factor)

        # Light noise
        if random.random() > 0.8:
            img_array = np.array(image)
            noise = np.random.normal(0, 2, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        return image, boxes

    def validate_box(self, box: List[float]) -> bool:
        """Validate box coordinates"""
        cls, x_center, y_center, width, height = box
        
        # Check bounds
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        if not (0 < width <= 1 and 0 < height <= 1):
            return False
        
        # Check size constraints
        if width < 0.02 or height < 0.02:  # Too small
            return False
        if width > 0.95 or height > 0.95:  # Too large
            return False
            
        return True

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".jpg", ".txt"))

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            img = Image.new('RGB', (self.img_size, self.img_size), color='black')

        # Load and parse labels
        boxes = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x_center, y_center, width, height = map(float, parts)
                            box = [cls, x_center, y_center, width, height]
                            
                            if self.validate_box(box):
                                boxes.append(box)
            except Exception as e:
                print(f"Error loading labels {label_path}: {e}")
        
        boxes = np.array(boxes) if boxes else np.zeros((0, 5))

        # Apply augmentations
        img, boxes = self.augment_image_and_boxes(img, boxes)

        # Resize image
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)

        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Initialize targets
        targets = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 5))
        
        for box in boxes:
            cls, x_center, y_center, width, height = box
            
            # Calculate grid coordinates
            grid_x = int(np.clip(x_center * self.grid_size, 0, self.grid_size - 1))
            grid_y = int(np.clip(y_center * self.grid_size, 0, self.grid_size - 1))
            
            # Find best anchor
            box_wh = [width * self.grid_size, height * self.grid_size]
            anchor_idx = self.find_best_anchor(box_wh)
            
            # Assign target if cell is empty
            if targets[anchor_idx, grid_y, grid_x, 4] == 0:
                # Relative coordinates
                x_rel = np.clip((x_center * self.grid_size) - grid_x, 0.01, 0.99)
                y_rel = np.clip((y_center * self.grid_size) - grid_y, 0.01, 0.99)
                
                # Log-space encoding
                w_rel = np.log(np.maximum(width * self.grid_size / self.anchors[anchor_idx, 0], 1e-8))
                h_rel = np.log(np.maximum(height * self.grid_size / self.anchors[anchor_idx, 1], 1e-8))
                
                targets[anchor_idx, grid_y, grid_x] = torch.tensor([
                    x_rel, y_rel, w_rel, h_rel, 1.0
                ])

        return img, targets

    def get_anchors(self) -> np.ndarray:
        return self.anchors


def compute_anchors(car_samples, config, num_anchors: int = 5) -> np.ndarray:
    """Compute anchors using k-means clustering"""
    bbox_sizes = []
    
    print("Computing improved anchors...")
    
    for img, tgt in car_samples:
        img_width, img_height = img.size
        
        # Skip small images
        if img_width < 64 or img_height < 64:
            continue
            
        for ann in tgt:
            x_min, y_min, width, height = ann['bbox']
            
            # Validate bbox
            if width <= 0 or height <= 0:
                continue
            if x_min < 0 or y_min < 0 or x_min + width > img_width or y_min + height > img_height:
                continue
            
            # Normalize and filter
            norm_width = width / img_width
            norm_height = height / img_height
            
            if norm_width < 0.02 or norm_height < 0.02:
                continue
            if norm_width > 0.95 or norm_height > 0.95:
                continue
            
            # Scale to grid
            grid_width = norm_width * config.grid_size
            grid_height = norm_height * config.grid_size
            
            bbox_sizes.append([grid_width, grid_height])
    
    if len(bbox_sizes) < num_anchors:
        print(f"Warning: Only {len(bbox_sizes)} valid boxes found, using fallback anchors")
        return np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                        [9.47112, 4.84053], [11.2364, 10.0071]])
    
    bbox_sizes = np.array(bbox_sizes)
    print(f"Extracted {len(bbox_sizes)} valid bounding box sizes")
    print(f"Size statistics - Mean: {bbox_sizes.mean(axis=0):.3f}, Std: {bbox_sizes.std(axis=0):.3f}")
    
    # K-means with multiple initializations
    best_anchors = None
    best_inertia = float('inf')
    
    for init in range(10):
        kmeans = KMeans(n_clusters=num_anchors, random_state=init, n_init=10).fit(bbox_sizes)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_anchors = kmeans.cluster_centers_
    
    # Sort by area
    areas = best_anchors[:, 0] * best_anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = best_anchors[sorted_indices]
    
    print(f"Computed anchors: {anchors}")
    return anchors


def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    paths = config.get_paths()
    root_dir = paths['parsed_dataset']
    
    # Create datasets
    train_dataset = ParsedYOLODataset(root_dir, "train", config, transform=transform, augment=True)
    val_dataset = ParsedYOLODataset(root_dir, "val", config, transform=transform, augment=False)
    
    # Check if test split exists
    test_dir = os.path.join(root_dir, "test")
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = ParsedYOLODataset(root_dir, "test", config, transform=transform, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0) if test_dataset else None
    
    print(f"Created data loaders:")
    print(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  - Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    if test_loader:
        print(f"  - Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader