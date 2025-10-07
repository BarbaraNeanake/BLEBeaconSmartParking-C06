import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.models import resnet50
import cv2

# YOLOv2-ResNet Model Definition (same as in train.py)
class YOLOv2ResNet(nn.Module):
    def __init__(self, num_anchors=5, num_classes=1):
        super(YOLOv2ResNet, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = resnet50(pretrained=False)  # Set to False since we're loading trained weights
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
num_anchors = 5
num_classes = 1
img_size = 416
grid_size = 13

# Load the trained model
model = YOLOv2ResNet(num_anchors=num_anchors, num_classes=num_classes)

# Load model weights (adjust path to your saved model)
model_path = "best_yolov2_resnet_car.pth"  # or "detector_car.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print(f"Model file {model_path} not found! Please check the path.")
    exit()

model = model.to(device)
model.eval()

# Load anchors
anchors_path = os.path.join("datasets", "COCO_car", "anchors.npy")
if os.path.exists(anchors_path):
    anchors = np.load(anchors_path)
    print(f"Anchors loaded: {anchors}")
else:
    # Fallback anchors if file not found
    anchors = np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                       [9.47112, 4.84053], [11.2364, 10.0071]])
    print("Using fallback anchors")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess image for PyTorch model
def preprocess_image(image_path):
    """Load and preprocess image for the model"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

# Decode YOLO predictions
def decode_predictions(predictions, anchors, conf_threshold=0.5, nms_threshold=0.4):
    """
    Decode YOLO predictions into bounding boxes
    predictions: [batch_size, num_anchors*(5+num_classes), grid_size, grid_size]
    """
    batch_size = predictions.size(0)
    grid_size = predictions.size(2)
    num_anchors = len(anchors)
    
    # Reshape predictions: [batch, num_anchors, grid, grid, 5+num_classes]
    predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
    
    # Apply transformations
    pred_xy = torch.sigmoid(predictions[..., :2])  # x,y coordinates [0,1]
    pred_wh = predictions[..., 2:4]  # w,h in log space
    pred_conf = torch.sigmoid(predictions[..., 4])  # confidence [0,1]
    pred_cls = torch.sigmoid(predictions[..., 5:]) if num_classes > 0 else None
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(num_anchors):
                    confidence = pred_conf[b, k, i, j].item()
                    
                    if confidence >= conf_threshold:
                        # Calculate box coordinates
                        x_center = (j + pred_xy[b, k, i, j, 0].item()) / grid_size
                        y_center = (i + pred_xy[b, k, i, j, 1].item()) / grid_size
                        
                        # Convert from log space and scale by anchors
                        w = torch.exp(pred_wh[b, k, i, j, 0]).item() * anchors[k, 0] / grid_size
                        h = torch.exp(pred_wh[b, k, i, j, 1]).item() * anchors[k, 1] / grid_size
                        
                        # Convert to corner coordinates
                        x_min = x_center - w / 2
                        y_min = y_center - h / 2
                        x_max = x_center + w / 2
                        y_max = y_center + h / 2
                        
                        # Clamp to [0, 1]
                        x_min = max(0, min(1, x_min))
                        y_min = max(0, min(1, y_min))
                        x_max = max(0, min(1, x_max))
                        y_max = max(0, min(1, y_max))
                        
                        all_boxes.append([x_min, y_min, x_max, y_max])
                        all_scores.append(confidence)
                        all_classes.append(0)  # Car class
    
    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy arrays
    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    classes = np.array(all_classes)
    
    # Apply Non-Maximum Suppression
    if len(boxes) > 0:
        # Convert to format expected by cv2.dnn.NMSBoxes: [x, y, w, h]
        boxes_nms = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            boxes_nms.append([x_min, y_min, w, h])
        
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores[indices], classes[indices]
    
    return boxes, scores, classes

# Visualize predictions
def visualize_predictions(image_path, boxes, scores, classes, output_path):
    """Visualize bounding box predictions on the image"""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        x_min, y_min, x_max, y_max = box
        
        # Convert normalized coordinates to pixel coordinates
        x_min_px = x_min * original_size[0]
        y_min_px = y_min * original_size[1]
        x_max_px = x_max * original_size[0]
        y_max_px = y_max * original_size[1]
        
        width = x_max_px - x_min_px
        height = y_max_px - y_min_px
        
        # Create rectangle
        rect = patches.Rectangle(
            (x_min_px, y_min_px), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f'Car: {score:.2f}'
        ax.text(
            x_min_px, y_min_px - 10, label,
            color='red', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
    
    ax.set_title(f'Car Detection Results - {len(boxes)} cars detected', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to: {output_path}")

# Test the model on images
def test_model():
    """Test the trained model on images"""
    
    # Create test directories
    test_dir = "test_images"
    results_dir = "test_results"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Looking for test images in: {test_dir}")
    print(f"Results will be saved to: {results_dir}")
    
    # Get test images
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not test_images:
        print(f"No test images found in {test_dir} folder.")
        print("Please add some images to test and rerun.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test each image
    for img_file in test_images:
        print(f"\n--- Testing {img_file} ---")
        image_path = os.path.join(test_dir, img_file)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(image_path)
            processed_image = processed_image.to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = model(processed_image)
            
            # Decode predictions
            boxes, scores, classes = decode_predictions(
                predictions.cpu(), anchors, conf_threshold=0.3, nms_threshold=0.4
            )
            
            print(f"Detected {len(boxes)} cars")
            if len(scores) > 0:
                print(f"Confidence scores: {scores}")
                print(f"Average confidence: {np.mean(scores):.3f}")
            
            # Save visualization
            output_filename = f"result_{img_file}"
            output_path = os.path.join(results_dir, output_filename)
            visualize_predictions(image_path, boxes, scores, classes, output_path)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"\n--- Testing Complete ---")
    print(f"Check the '{results_dir}' folder for visualization results")

# Run the test
if __name__ == "__main__":
    test_model()