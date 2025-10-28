# data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ParsedYOLODataset(Dataset):
    def __init__(self, root_dir, split, transform=None, img_size=416):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.label_dir = os.path.join(root_dir, split, "labels")
        self.transform = transform
        self.img_size = img_size
        self.anchors = np.load(os.path.join(os.getcwd(), "datasets", "COCO_car", "anchors.npy"))
        self.num_anchors = len(self.anchors)
        self.grid_size = img_size // 32  # 416 // 32 = 13 for YOLOv2

        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        self.image_files.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".jpg", ".txt"))

        # Load image
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size

        # Resize image
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)

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
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

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
            
            # For simplicity, assign to first anchor (you can improve this with IoU matching)
            anchor_idx = 0
            
            # Calculate relative coordinates within the grid cell
            x_rel = x_center * self.grid_size - grid_x
            y_rel = y_center * self.grid_size - grid_y
            
            # Convert width and height to grid scale
            w_scaled = width * self.grid_size
            h_scaled = height * self.grid_size
            
            # Assign target values [x_rel, y_rel, w_scaled, h_scaled, confidence]
            targets[anchor_idx, grid_y, grid_x] = torch.tensor([
                x_rel, y_rel, w_scaled, h_scaled, 1.0
            ])
        
        # print(f"Per-sample targets shape: {targets.shape}")  # Debug per-sample shape

        return img, targets

    def get_anchors(self):
        return self.anchors