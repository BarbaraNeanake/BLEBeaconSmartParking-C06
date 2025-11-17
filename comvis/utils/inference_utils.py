"""
Inference utilities for SPARK car detection pipeline
"""
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import cv2
from typing import Tuple, List, Optional
from .model import YOLOv2ResNet, create_model


class ModelInference:
    """
    Model inference with improved post-processing and visualization
    """
    
    def __init__(self, config, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.transform = None
        self.anchors = None
        
        # Get paths
        self.paths = config.get_paths()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load anchors
        self._load_anchors()
    
    def _load_anchors(self) -> None:
        """Load anchor boxes"""
        anchors_path = self.paths['anchors_file']
        if os.path.exists(anchors_path):
            self.anchors = np.load(anchors_path)
            print(f"✓ Loaded anchors: {self.anchors}")
        else:
            # Fallback anchors
            self.anchors = np.array([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
                                   [9.47112, 4.84053], [11.2364, 10.0071]])
            print("⚠ Using fallback anchors")
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load trained model"""
        if model_path is None:
            # Try to find best model
            model_paths = [
                self.paths['best_model'],
                self.paths['final_model'],
                'detector_car2.pth',  # Legacy
                'best_yolov2_resnet_car.pth'  # Legacy
            ]
        else:
            model_paths = [model_path]
        
        # Create model
        self.model = create_model(self.config, self.device, pretrained=False)
        
        # Try to load weights
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.model.load_pretrained_weights(path, self.device)
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        if not model_loaded:
            raise FileNotFoundError("No compatible model found! Please train the model first.")
        
        self.model.eval()
        print("✓ Model loaded and ready for inference")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Preprocess image for inference"""
        # Load original image
        original_img = Image.open(image_path).convert("RGB")
        
        # Resize for model
        img = original_img.resize((self.config.img_size, self.config.img_size), Image.BILINEAR)
        img_tensor = self.transform(img)
        
        return img_tensor.unsqueeze(0), original_img
    
    def decode_predictions(self, predictions: torch.Tensor, 
                          conf_threshold: Optional[float] = None,
                          nms_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode YOLO predictions with improved coordinate handling
        """
        if conf_threshold is None:
            conf_threshold = self.config.conf_threshold
        if nms_threshold is None:
            nms_threshold = self.config.nms_threshold
        
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        num_anchors = len(self.anchors)
        
        # Reshape predictions
        predictions = predictions.view(batch_size, num_anchors, 5 + self.config.num_classes, grid_size, grid_size)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        # Apply transformations
        pred_xy = torch.sigmoid(predictions[..., :2])
        pred_wh = predictions[..., 2:4]
        pred_conf = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:]) if self.config.num_classes > 0 else None
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Create grid coordinates
        grid_x, grid_y = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='xy')
        grid_x = grid_x.float().to(predictions.device)
        grid_y = grid_y.float().to(predictions.device)
        
        for b in range(batch_size):
            batch_boxes = []
            batch_scores = []
            batch_classes = []
            
            for k in range(num_anchors):
                # Get confidence mask
                conf_mask = pred_conf[b, k] >= conf_threshold
                
                if conf_mask.sum() == 0:
                    continue
                
                # Extract valid predictions
                xy_pred = pred_xy[b, k][conf_mask]
                wh_pred = pred_wh[b, k][conf_mask]
                conf_pred = pred_conf[b, k][conf_mask]
                
                # Get grid coordinates
                grid_x_valid = grid_x[conf_mask]
                grid_y_valid = grid_y[conf_mask]
                
                # Calculate box coordinates
                x_center = (grid_x_valid + xy_pred[:, 0]) / grid_size
                y_center = (grid_y_valid + xy_pred[:, 1]) / grid_size
                
                # Convert from log space (simple, no fancy clipping)
                anchor_w = self.anchors[k, 0] / grid_size
                anchor_h = self.anchors[k, 1] / grid_size
                
                w = torch.exp(torch.clamp(wh_pred[:, 0], -4, 4)) * anchor_w
                h = torch.exp(torch.clamp(wh_pred[:, 1], -4, 4)) * anchor_h
                
                # Convert to corner coordinates
                x_min = torch.clamp(x_center - w / 2, 0, 1)
                y_min = torch.clamp(y_center - h / 2, 0, 1)
                x_max = torch.clamp(x_center + w / 2, 0, 1)
                y_max = torch.clamp(y_center + h / 2, 0, 1)
                
                # Simple filter: just check boxes are valid
                valid_mask = (x_max > x_min) & (y_max > y_min)
                
                if valid_mask.sum() > 0:
                    boxes = torch.stack([x_min[valid_mask], y_min[valid_mask], 
                                       x_max[valid_mask], y_max[valid_mask]], dim=1)
                    scores = conf_pred[valid_mask]
                    classes = torch.zeros_like(scores)
                    
                    batch_boxes.append(boxes.cpu().numpy())
                    batch_scores.append(scores.cpu().numpy())
                    batch_classes.append(classes.cpu().numpy())
            
            if batch_boxes:
                all_boxes.append(np.concatenate(batch_boxes, axis=0))
                all_scores.append(np.concatenate(batch_scores, axis=0))
                all_classes.append(np.concatenate(batch_classes, axis=0))
            else:
                all_boxes.append(np.array([]).reshape(0, 4))
                all_scores.append(np.array([]))
                all_classes.append(np.array([]))
        
        # Apply NMS for first batch
        if len(all_boxes) > 0 and len(all_boxes[0]) > 0:
            boxes = all_boxes[0]
            scores = all_scores[0]
            classes = all_classes[0]
            
            # Convert to NMS format
            boxes_nms = []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                boxes_nms.append([x_min, y_min, w, h])
            
            if len(boxes_nms) > 0:
                indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), conf_threshold, nms_threshold)
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    return boxes[indices], scores[indices], classes[indices]
        
        return np.array([]).reshape(0, 4), np.array([]), np.array([])
    
    def visualize_predictions(self, image_path: str, boxes: np.ndarray, scores: np.ndarray, 
                            classes: np.ndarray, output_path: str) -> None:
        """Visualize predictions with improved rendering"""
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x_min, y_min, x_max, y_max = box
            
            # Convert to pixel coordinates
            x_min_px = x_min * original_size[0]
            y_min_px = y_min * original_size[1]
            x_max_px = x_max * original_size[0]
            y_max_px = y_max * original_size[1]
            
            width = x_max_px - x_min_px
            height = y_max_px - y_min_px
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x_min_px, y_min_px), width, height,
                linewidth=3, edgecolor='red', facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            label = f'Car: {score:.3f}'
            ax.text(
                x_min_px, y_min_px - 15, label,
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.8)
            )
            
            # Add corner markers
            corner_size = 8
            for corner_x, corner_y in [(x_min_px, y_min_px), (x_max_px, y_min_px), 
                                      (x_min_px, y_max_px), (x_max_px, y_max_px)]:
                circle = patches.Circle((corner_x, corner_y), corner_size, 
                                      color='red', alpha=0.8)
                ax.add_patch(circle)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=200, facecolor='white')
        plt.close()
        print(f"✓ Visualization saved: {output_path}")
    
    def predict_single_image(self, image_path: str, save_result: bool = True,
                           conf_threshold: Optional[float] = None,
                           nms_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        img_tensor, original_img = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Decode
        boxes, scores, classes = self.decode_predictions(
            predictions.cpu(), conf_threshold, nms_threshold
        )
        
        # Save result if requested
        if save_result and len(boxes) > 0:
            os.makedirs(self.paths['results_dir'], exist_ok=True)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(self.paths['results_dir'], f'result_{image_name}.jpg')
            self.visualize_predictions(image_path, boxes, scores, classes, output_path)
        
        return boxes, scores, classes
    
    def test_on_images(self, test_images_dir: Optional[str] = None,
                      conf_threshold: Optional[float] = None,
                      nms_threshold: Optional[float] = None) -> None:
        """Test model on multiple images"""
        if test_images_dir is None:
            test_images_dir = self.paths['test_images_dir']
        
        if not os.path.exists(test_images_dir):
            print(f"⚠ Test images directory not found: {test_images_dir}")
            return
        
        # Get test images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = [f for f in os.listdir(test_images_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if not test_images:
            print(f"⚠ No test images found in {test_images_dir}")
            return
        
        print(f"Testing on {len(test_images)} images...")
        os.makedirs(self.paths['results_dir'], exist_ok=True)
        
        total_detections = 0
        for img_file in test_images:
            print(f"\n--- Testing {img_file} ---")
            image_path = os.path.join(test_images_dir, img_file)
            
            try:
                boxes, scores, classes = self.predict_single_image(
                    image_path, save_result=True, 
                    conf_threshold=conf_threshold, 
                    nms_threshold=nms_threshold
                )
                
                print(f"Detected {len(boxes)} cars")
                if len(scores) > 0:
                    print(f"Confidence scores: {[f'{s:.3f}' for s in scores]}")
                    print(f"Average confidence: {np.mean(scores):.3f}")
                    total_detections += len(boxes)
                
            except Exception as e:
                print(f"✗ Error processing {img_file}: {str(e)}")
        
        print(f"\n{'='*50}")
        print("Testing Complete!")
        print(f"{'='*50}")
        print(f"Total images processed: {len(test_images)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(test_images):.1f}")
        print(f"Results saved to: {self.paths['results_dir']}")


def run_inference(config, model_path: Optional[str] = None, 
                 test_images_dir: Optional[str] = None,
                 device: str = 'cpu') -> ModelInference:
    """
    Convenience function to run inference with given configuration
    
    Args:
        config: Configuration object
        model_path: Path to trained model (optional)
        test_images_dir: Directory with test images (optional)
        device: Inference device
        
    Returns:
        ModelInference instance
    """
    inference = ModelInference(config, device)
    inference.load_model(model_path)
    inference.test_on_images(test_images_dir)
    return inference