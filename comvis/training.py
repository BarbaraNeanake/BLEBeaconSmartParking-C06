import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.cluster import KMeans

#%%
# Dataset Merging with FiftyOne
def merge_datasets(pklot_dir="PKLot", coco_max_samples=12000, output_dir="merged_dataset"):
    print(f"Starting dataset merging with PKLot dir: {pklot_dir}, output dir: {output_dir}")

    # Load PKLot dataset (VOC format, filter for cars)
    try:
        pklot_dataset = fo.Dataset.from_dir(
            dataset_dir=pklot_dir,
            dataset_type=fo.types.VOCDetectionDataset,
            label_field="detections",
        )
        print(f"Loaded PKLot dataset with {len(pklot_dataset)} samples")
        pklot_view = pklot_dataset.filter_labels(
            "detections",
            fo.ViewField("label") == "car"
        )
        print(f"Filtered PKLot dataset to {len(pklot_view)} car samples")
    except Exception as e:
        raise ValueError(f"Failed to load PKLot dataset from {pklot_dir}: {str(e)}")

    # Load COCO dataset (car class only)
    try:
        coco_dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            classes=["car"],
            label_field="detections",
            max_samples=coco_max_samples
        )
        print(f"Loaded COCO dataset with {len(coco_dataset)} car samples")
    except Exception as e:
        raise ValueError(f"Failed to load COCO dataset: {str(e)}")

    # Merge datasets
    merged_dataset = fo.Dataset(name="pklot_coco_merged")
    for sample in pklot_view:
        merged_dataset.add_sample(sample)
    for sample in coco_dataset:
        merged_dataset.add_sample(sample)
    print(f"Merged dataset has {len(merged_dataset)} samples")

    # Shuffle and manually split into train (80%), val (10%), test (10%)
    total_samples = len(merged_dataset)
    if total_samples == 0:
        raise ValueError("Merged dataset is empty. Check PKLot and COCO data sources.")
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    print(f"Splitting: train={train_size}, val={val_size}, test={test_size}")

    # Shuffle dataset
    shuffled_view = merged_dataset.shuffle(seed=42)

    # Assign splits using tags
    for idx, sample in enumerate(shuffled_view):
        if idx < train_size:
            sample.tags.append("train")
        elif idx < train_size + val_size:
            sample.tags.append("val")
        else:
            sample.tags.append("test")
        sample.save()

    # Export to YOLO format for each split
    for split in ["train", "val", "test"]:
        split_view = merged_dataset.match_tags(split)
        if len(split_view) == 0:
            raise ValueError(f"No samples in {split} split. Check dataset merging and splitting.")
        export_path = os.path.join(output_dir, split)
        print(f"Exporting {split} split with {len(split_view)} samples to {export_path}")
        split_view.export(
            export_dir=export_path,
            dataset_type=fo.types.YOLOv5Dataset,
            classes=["car"]
        )

    # Verify export
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, split, "images")
        label_dir = os.path.join(output_dir, split, "labels")
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise ValueError(f"Export failed: {img_dir} or {label_dir} does not exist")
        img_count = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        label_count = len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
        print(f"{split} split: {img_count} images, {label_count} labels")
        if img_count == 0 or label_count == 0:
            raise ValueError(f"No images or labels in {split} split")

    # Compute anchor priors
    bbox_sizes = []
    for sample in merged_dataset:
        for det in sample.detections.detections:
            _, _, w, h = det.bounding_box
            bbox_sizes.append([w * 13, h * 13])  # Scale to 13x13 grid
    if not bbox_sizes:
        raise ValueError("No bounding boxes found for anchor computation")
    bbox_sizes = np.array(bbox_sizes)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(bbox_sizes)
    anchors = kmeans.cluster_centers_
    print(f"Computed anchors: {anchors}")
    return anchors

#%%
# Dataset
class MergedYOLODataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train", anchors=None):
        self.root_dir = f"{root_dir}/{split}"
        self.transform = transform
        self.image_paths = []
        self.annotations = []
        
        # Load image and label paths
        img_dir = f"{self.root_dir}/images"
        label_dir = f"{self.root_dir}/labels"
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise ValueError(f"Directory missing for {split} split: {img_dir} or {label_dir}")
        for img_file in os.listdir(img_dir):
            if img_file.endswith(".jpg"):
                img_path = f"{img_dir}/{img_file}"
                label_path = f"{label_dir}/{img_file.replace('.jpg', '.txt')}"
                if os.path.exists(label_path):
                    self.image_paths.append(img_path)
                    self.annotations.append(label_path)
        
        if not self.image_paths:
            raise ValueError(f"No valid image-label pairs found in {self.root_dir}")
        print(f"Loaded {split} dataset with {len(self.image_paths)} samples")

        # Anchor priors
        self.anchors = anchors if anchors is not None else np.array([
            [1.5, 2.0], [3.0, 4.0], [4.5, 6.0], [6.0, 8.0], [8.0, 10.0]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.annotations[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load bounding boxes from YOLO text file
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    if cls == 0:  # "car" class
                        x_min = x - w / 2
                        y_min = y - h / 2
                        x_max = x + w / 2
                        y_max = y + h / 2
                        boxes.append([x_min, y_min, x_max, y_max, 1.0])  # [x_min, y_min, x_max, y_max, class]
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        targets = self.to_yolo_targets(boxes)
        return image, targets

    def to_yolo_targets(self, boxes):
        grid_size = 13
        num_anchors = 5
        targets = torch.zeros((grid_size, grid_size, num_anchors, 6))  # [x, y, w, h, obj, cls]
        
        for box in boxes:
            x_min, y_min, x_max, y_max, cls = box
            w = x_max - x_min
            h = y_max - y_min
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            
            grid_x = int(x * grid_size)
            grid_y = int(y * grid_size)
            if grid_x >= grid_size or grid_y >= grid_size:
                continue
            
            anchor_ious = []
            for anchor in self.anchors:
                anchor_w, anchor_h = anchor
                iou = min(w, anchor_w) * min(h, anchor_h) / max(w, anchor_w) / max(h, anchor_h)
                anchor_ious.append(iou)
            anchor_idx = np.argmax(anchor_ious)
            
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
        self.conv3 = nn.Conv2d(1088, 30, 1, 1, 0)

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
            if i == 24:
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
        return out

#%%
# Loss Function
def yolo_loss(preds, targets, anchors, grid_size=13, lambda_coord=5.0, lambda_noobj=0.5):
    batch_size = preds.size(0)
    num_anchors = anchors.shape[0]
    mse_loss = nn.MSELoss(reduction='sum')
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    
    preds = preds.view(batch_size, grid_size, grid_size, num_anchors, 6)
    
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
                    obj_mask = target[4] > 0.5
                    if obj_mask:
                        coord_loss += lambda_coord * mse_loss(pred[:4], target[:4])
                        obj_loss += bce_loss(pred[4], target[4])
                        class_loss += bce_loss(pred[5], target[5])
                    else:
                        noobj_loss += lambda_noobj * bce_loss(pred[4], target[4])
    
    total_loss = (coord_loss + obj_loss + noobj_loss + class_loss) / batch_size
    return total_loss, {"coord": coord_loss / batch_size, 
                        "obj": obj_loss / batch_size, 
                        "noobj": noobj_loss / batch_size, 
                        "class": class_loss / batch_size}

#%%
# Training Loop
def train_model(epochs=50, batch_size=16, lr=1e-3, pklot_dir="PKLot", output_dir="merged_dataset"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Merge datasets and compute anchors
    anchors = merge_datasets(pklot_dir=pklot_dir, output_dir=output_dir)
    print("Computed Anchors:", anchors)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    
    # Load datasets
    train_dataset = MergedYOLODataset(root_dir=output_dir, transform=transform, split="train", anchors=anchors)
    val_dataset = MergedYOLODataset(root_dir=output_dir, transform=transform, split="val", anchors=anchors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and optimizer
    model = Darknet19(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    
    anchors = torch.tensor(anchors, dtype=torch.float32).to(device)
    
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
    
    # Save model
    torch.save(model.state_dict(), 'yolo_v2.pth')
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez('weights.npz', **weights)

#%%
# Run Training
if __name__ == "__main__":
    train_model(epochs=50, batch_size=16, lr=1e-3, pklot_dir="PKLot", output_dir="merged_dataset")
# %%
fo.delete_dataset('pklot_coco_merged')