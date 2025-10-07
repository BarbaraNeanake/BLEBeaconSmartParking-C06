# data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class ParsedYOLODataset(Dataset):
    def __init__(self, root_dir, split, transform=None, img_size=416, augment=True):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.label_dir = os.path.join(root_dir, split, "labels")
        self.transform = transform
        self.img_size = img_size
        self.augment = augment and split == "train"  # Only augment training data
        self.anchors = np.load(os.path.join(os.getcwd(), "datasets", "COCO_car", "anchors.npy"))
        self.num_anchors = len(self.anchors)
        self.grid_size = img_size // 32  # 416 // 32 = 13 for YOLOv2

        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.image_files.sort()  # Ensure consistent ordering

    def find_best_anchor(self, box_wh):
        """Find the best anchor for a given box using IoU"""
        box_wh = np.array(box_wh)
        anchor_wh = self.anchors
        
        # Calculate IoU between box and each anchor
        intersection = np.minimum(box_wh[0], anchor_wh[:, 0]) * np.minimum(box_wh[1], anchor_wh[:, 1])
        box_area = box_wh[0] * box_wh[1]
        anchor_areas = anchor_wh[:, 0] * anchor_wh[:, 1]
        union = box_area + anchor_areas - intersection
        
        iou = intersection / union
        return np.argmax(iou)

    def augment_image_and_boxes(self, image, boxes):
        """Apply data augmentation to image and adjust bounding boxes accordingly"""
        if not self.augment or len(boxes) == 0:
            return image, boxes

        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # Flip x coordinates
            boxes[:, 1] = 1.0 - boxes[:, 1]  # x_center = 1 - x_center

        # Random brightness/contrast adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_brightness(image, brightness_factor)

        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = transforms.functional.adjust_contrast(image, contrast_factor)

        return image, boxes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".jpg", ".txt"))

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size

        # Load and parse labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # [class, x_center, y_center, width, height]
                        cls, x_center, y_center, width, height = map(float, parts)
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                            boxes.append([cls, x_center, y_center, width, height])
        
        boxes = np.array(boxes) if boxes else np.zeros((0, 5))

        # Apply augmentations
        img, boxes = self.augment_image_and_boxes(img, boxes)

        # Resize image
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)

        # Convert boxes to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)

        # Initialize target tensor [num_anchors, grid_size, grid_size, 5]
        targets = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 5))
        
        for box in boxes:
            cls, x_center, y_center, width, height = box
            
            # Calculate grid cell coordinates
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)
            
            # Clamp to valid grid range
            grid_x = max(0, min(grid_x, self.grid_size - 1))
            grid_y = max(0, min(grid_y, self.grid_size - 1))
            
            # Find best anchor using IoU
            box_wh = [width * self.grid_size, height * self.grid_size]
            anchor_idx = self.find_best_anchor(box_wh)
            
            # Check if this cell already has an object for this anchor
            if targets[anchor_idx, grid_y, grid_x, 4] == 0:  # No object yet
                # Calculate relative coordinates within the grid cell
                x_rel = x_center * self.grid_size - grid_x  # [0, 1]
                y_rel = y_center * self.grid_size - grid_y  # [0, 1]
                
                # Store width/height relative to anchor size for log-space encoding
                w_rel = np.log(width * self.grid_size / self.anchors[anchor_idx, 0] + 1e-8)
                h_rel = np.log(height * self.grid_size / self.anchors[anchor_idx, 1] + 1e-8)
                
                # Assign target values [x_rel, y_rel, w_log, h_log, confidence]
                targets[anchor_idx, grid_y, grid_x] = torch.tensor([
                    x_rel, y_rel, w_rel, h_rel, 1.0
                ])
        
        # print(f"Per-sample targets shape: {targets.shape}")  # Debug per-sample shape

        return img, targets

    def get_anchors(self):
        return self.anchors