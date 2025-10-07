#%% train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data import ParsedYOLODataset
from torchvision.models import resnet50

#%% Define train_transform (assuming ImageNet normalization for ResNet50)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom YOLOv2 head on top of ResNet50
class YOLOv2ResNet(nn.Module):
    def __init__(self, num_anchors=5, num_classes=1):
        super(YOLOv2ResNet, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = resnet50(pretrained=True)
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # Add custom YOLOv2 detection head
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        return x

#%% Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_epochs = 50  # Increased for better convergence
batch_size = 8
learning_rate = 0.001
img_size = 416
grid_size = img_size // 32  # 13
num_classes = 1  # Only "car" class
num_anchors = 5

# Initialize model
model = YOLOv2ResNet(num_anchors=num_anchors, num_classes=num_classes)
model = model.to(device)

# Freeze backbone layers initially (optional, unfreeze later)
for param in model.backbone.parameters():
    param.requires_grad = False

# Loss function (improved YOLO loss)
class YOLOLoss(nn.Module):
    def __init__(self, anchors, device, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32).to(device)
        self.device = device
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        num_anchors = self.anchors.size(0)
        num_classes = predictions.size(1) // num_anchors - 5

        # Reshape predictions: [batch, num_anchors*(5+classes), grid, grid] -> [batch, num_anchors, grid, grid, 5+classes]
        predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        # Apply proper YOLO transformations
        pred_xy = torch.sigmoid(predictions[..., :2])  # x,y should be sigmoid [0,1]
        pred_wh = predictions[..., 2:4]  # w,h will be exponential with anchors
        pred_conf = torch.sigmoid(predictions[..., 4])  # confidence sigmoid
        pred_cls = torch.sigmoid(predictions[..., 5:]) if num_classes > 0 else None

        # Ensure targets have the same shape as predictions
        if targets.dim() == 4:
            targets = targets.view(batch_size, num_anchors, grid_size, grid_size, 5)

        # Object mask
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask

        # Coordinate loss (only for cells with objects)
        if obj_mask.sum() > 0:
            # Extract target coordinates (already in the right format)
            target_xy = targets[..., :2][obj_mask]  # [0,1] relative to grid cell
            target_wh = targets[..., 2:4][obj_mask]  # log-space relative to anchors
            
            pred_xy_obj = pred_xy[obj_mask]
            pred_wh_obj = pred_wh[obj_mask]
            
            # XY loss - both are in [0,1] range
            xy_loss = self.mse_loss(pred_xy_obj, target_xy).sum()
            
            # WH loss - both are in log-space
            wh_loss = self.mse_loss(pred_wh_obj, target_wh).sum()
            
            coord_loss = xy_loss + wh_loss
        else:
            coord_loss = torch.tensor(0.0, device=self.device)

        # Confidence loss
        obj_pred = pred_conf[obj_mask]
        obj_target = targets[..., 4][obj_mask]
        noobj_pred = pred_conf[noobj_mask]
        noobj_target = torch.zeros_like(noobj_pred)

        obj_loss = self.bce_loss(obj_pred, obj_target).sum() if obj_pred.numel() > 0 else torch.tensor(0.0, device=self.device)
        noobj_loss = self.bce_loss(noobj_pred, noobj_target).sum() if noobj_pred.numel() > 0 else torch.tensor(0.0, device=self.device)

        # Class loss
        if num_classes > 0 and obj_mask.sum() > 0:
            class_pred = pred_cls[obj_mask]
            class_target = torch.ones_like(class_pred)  # For single class
            class_loss = self.bce_loss(class_pred, class_target).sum()
        else:
            class_loss = torch.tensor(0.0, device=self.device)

        # Normalize by batch size and apply weights
        batch_norm = batch_size
        total_loss = (
            self.lambda_coord * coord_loss / batch_norm +
            obj_loss / batch_norm +
            self.lambda_noobj * noobj_loss / batch_norm +
            class_loss / batch_norm
        )

        return total_loss

# Dataset and DataLoader
root_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "parsed_dataset")
train_dataset = ParsedYOLODataset(root_dir, "train", transform=train_transform, img_size=img_size, augment=True)
val_dataset = ParsedYOLODataset(root_dir, "val", transform=train_transform, img_size=img_size, augment=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Optimizer and scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # Reduce LR every 15 epochs

# Load anchors
anchors = train_dataset.get_anchors()
criterion = YOLOLoss(anchors, device, lambda_coord=5.0, lambda_noobj=0.5)

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#%% Training loop
total_steps = len(train_loader)
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    # Step the scheduler
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, targets).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    current_lr = scheduler.get_last_lr()[0]
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_yolov2_resnet_car.pth')
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Unfreeze backbone after a few epochs (optional)
    if epoch == 5:  # Unfreeze earlier since we have more epochs
        print("Unfreezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=learning_rate * 0.1)  # Lower LR for full model
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("Training finished. Save model if needed:")
# torch.save(model.state_dict(), 'yolov2_resnet_car.pth')
# %%
torch.save(model.state_dict(), 'detector_car2.pth')
# %%
