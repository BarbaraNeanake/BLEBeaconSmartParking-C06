#%%
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torch.cuda.amp import GradScaler, autocast

#%%
# Dataset
class CNRParkDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.annotations = []
        self.camera_csvs = {f"camera{i}": pd.read_csv(f"{root_dir}/camera{i}.csv") for i in range(1, 10)}
        
        # Load image paths and annotations
        for weather in ["SUNNY", "OVERCAST", "RAINY"]:
            weather_dir = f"{root_dir}/{weather}"
            if not os.path.exists(weather_dir):
                continue
            for date_dir in os.listdir(weather_dir):
                date_path = f"{weather_dir}/{date_dir}"
                for cam_dir in os.listdir(date_path):
                    cam_id = cam_dir.replace("camera", "")
                    cam_path = f"{date_path}/{cam_dir}"
                    for img_file in os.listdir(cam_path):
                        self.image_paths.append(f"{cam_path}/{img_file}")
                        self.annotations.append((cam_id, img_file))
        
        # Split dataset (80% train, 10% val, 10% test)
        total = len(self.image_paths)
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        if split == "train":
            self.image_paths = self.image_paths[:train_end]
            self.annotations = self.annotations[:train_end]
        elif split == "val":
            self.image_paths = self.image_paths[train_end:val_end]
            self.annotations = self.annotations[train_end:val_end]
        else:  # test
            self.image_paths = self.image_paths[val_end:]
            self.annotations = self.annotations[val_end:]
        
        # Anchor priors (from k-means on CNRPark+EXT, placeholder values)
        self.anchors = np.array([
            [1.5, 2.0], [3.0, 4.0], [4.5, 6.0], [6.0, 8.0], [8.0, 10.0]
        ])  # Normalized to 13x13 grid

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        cam_id, img_file = self.annotations[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load bounding boxes
        boxes = []
        csv = self.camera_csvs[f"camera{cam_id}"]
        for _, row in csv.iterrows():
            if row['label'] == 1:  # "busy"
                x_min = row['x_min'] * 1000 / 2592
                y_min = row['y_min'] * 750 / 1944
                x_max = row['x_max'] * 1000 / 2592
                y_max = row['y_max'] * 750 / 1944
                boxes.append([x_min, y_min, x_max, y_max, 1.0])  # [x_min, y_min, x_max, y_max, class]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Convert to YOLO targets
        targets = self.to_yolo_targets(boxes)
        return image, targets

    def to_yolo_targets(self, boxes):
        grid_size = 13
        num_anchors = 5
        targets = torch.zeros((grid_size, grid_size, num_anchors, 6))  # [x, y, w, h, obj, cls]
        
        for box in boxes:
            x_min, y_min, x_max, y_max, cls = box
            w = (x_max - x_min) / 416  # Normalize to [0,1]
            h = (y_max - y_min) / 416
            x = (x_min + x_max) / 2 / 416
            y = (y_min + y_max) / 2 / 416
            
            # Assign to grid cell
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            if grid_x >= grid_size or grid_y >= grid_size:
                continue
            
            # Find best anchor
            anchor_ious = []
            for anchor in self.anchors:
                anchor_w, anchor_h = anchor
                iou = min(w, anchor_w) * min(h, anchor_h) / max(w, anchor_w) / max(h, anchor_h)
                anchor_ious.append(iou)
            anchor_idx = np.argmax(anchor_ious)
            
            # Assign target
            targets[grid_y, grid_x, anchor_idx] = torch.tensor([x, y, w, h, 1.0, cls])
        
        return targets

#%%
# Model
class Darknet19(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1, 1, 0), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, 1, 0), nn.BatchNorm2d(512), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True),
        )
        self.passthrough_conv = nn.Conv2d(512, 64, 1, 1, 0)
        self.conv1 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv3 = nn.Conv2d(1088, 30, 1, 1, 0)  # 5 anchors x (4 coords + 1 obj + 1 cls)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/marvis/pytorch-yolo2/releases/download/v0.0/darknet19.pth',
                progress=True
            )
            backbone_dict = {k: v for k, v in state_dict.items() if k.startswith('features')}
            self.features.load_state_dict(backbone_dict)

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 24:  # After 4th maxpool (26x26x512)
                passthrough = x
            features.append(x)
        passthrough = self.passthrough_conv(passthrough)
        passthrough = passthrough.view(passthrough.size(0), 64, passthrough.size(2)//2, 2, passthrough.size(3)//2, 2)
        passthrough = passthrough.permute(0, 1, 3, 5, 2, 4).contiguous()
        passthrough = passthrough.view(passthrough.size(0), 64*4, passthrough.size(2), passthrough.size(4))
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = torch.cat([passthrough, out], dim=1)
        out = self.conv3(out)
        return out  # (batch, 13, 13, 30)

#%%
# Loss Function
def yolo_loss(preds, targets, anchors, grid_size=13, lambda_coord=5.0, lambda_noobj=0.5):
    """
    YOLOv2 loss: MSE for coords, BCE for objectness/class, weighted.
    preds: (batch, 13, 13, 5*(4+1+1)) - [x,y,w,h,obj,cls]
    targets: (batch, 13, 13, 5, 6) - [x,y,w,h,obj,cls]
    anchors: (5, 2) - normalized to grid
    """
    batch_size = preds.size(0)
    num_anchors = anchors.shape[0]
    mse_loss = nn.MSELoss(reduction='sum')
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    # Reshape predictions
    preds = preds.view(batch_size, grid_size, grid_size, num_anchors, 6)  # [x,y,w,h,obj,cls]
    
    # Initialize losses
    coord_loss = 0.0
    obj_loss = 0.0
    noobj_loss = 0.0
    class_loss = 0.0
    
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                for a in range(num_anchors):
                    target = targets[b, i, j, a]
                    pred = preds[b, i, j, a]
                    
                    # Objectness indicator
                    obj_mask = target[4] > 0.5
                    
                    if obj_mask:
                        # Coordinate loss (x, y, w, h)
                        coord_loss += lambda_coord * mse_loss(pred[:4], target[:4])
                        
                        # Objectness loss
                        obj_loss += bce_loss(pred[4], target[4])
                        
                        # Class loss (single class: "car")
                        class_loss += bce_loss(pred[5], target[5])
                    else:
                        # No-object loss
                        noobj_loss += lambda_noobj * bce_loss(pred[4], target[4])
    
    total_loss = (coord_loss + obj_loss + noobj_loss + class_loss) / batch_size
    return total_loss, {"coord": coord_loss / batch_size, 
                        "obj": obj_loss / batch_size, 
                        "noobj": noobj_loss / batch_size, 
                        "class": class_loss / batch_size}

#%%
# Training Loop
def train_model(epochs=50, batch_size=16, lr=1e-3):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    train_dataset = CNRParkDataset(root_dir="CNRPark_EXT/full_images", transform=transform, split="train")
    val_dataset = CNRParkDataset(root_dir="CNRPark_EXT/full_images", transform=transform, split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = Darknet19(pretrained=True).to(device)
    
    # Optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    # Anchors (same as dataset)
    anchors = torch.tensor([
        [1.5, 2.0], [3.0, 4.0], [4.5, 6.0], [6.0, 8.0], [8.0, 10.0]
    ]).to(device)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with autocast():
                preds = model(images)
                loss, loss_dict = yolo_loss(preds, targets, anchors)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                with autocast():
                    preds = model(images)
                    loss, _ = yolo_loss(preds, targets, anchors)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save weights
    torch.save(model.state_dict(), 'yolo_v2.pth')
    
    # Export to NumPy for inference
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez('weights.npz', **weights)

#%%
# Run Training
if __name__ == "__main__":
    train_model(epochs=50, batch_size=16, lr=1e-3)