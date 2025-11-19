"""
Pure Python inference engine for YOLOv2-ResNet34
High-performance inference with NumPy
"""

import numpy as np
import json
import cv2
from typing import Dict, List, Tuple, Optional
from ._model import YOLOv2ResNet
from ._nms import nms


class SPARKEngine:
    """YOLOv2 inference engine"""
    
    def __init__(self, model_path: str, config_path: str, 
                 input_size: int = 416, backend: str = "resnet34",
                 anchor_path: str = "anchors.npy"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model weights (.pth file)
            config_path: Path to configuration JSON
            input_size: Input image size (default 416)
            backend: Backbone architecture (resnet34 only)
            anchor_path: Path to anchor file (.npy format, default "anchors.npy")
        """
        self.input_size = input_size
        self.num_classes = 1
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.class_names = ["car"]
        self.anchors = None
        
        # Load configuration
        self._load_config(config_path)
        
        # Load anchors from .npy file
        self._load_anchors(anchor_path)
        
        # Create model
        if backend == "resnet34":
            self.model = YOLOv2ResNet(self.num_classes, "resnet34")
        else:
            raise ValueError(f"Unsupported backend: {backend}. Only 'resnet34' is supported.")
        
        # Load weights
        self._load_weights(model_path)
        
        self.is_initialized = True
        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0
        }
    
    def _load_config(self, config_path: str):
        """Load model configuration from JSON"""
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                # Remove trailing commas before closing braces/brackets (common JSON error)
                content = content.replace(',}', '}').replace(',]', ']')
                config = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing error in {config_path}: {e}")
            print(f"  Error at line {e.lineno}, column {e.colno}")
            raise
        
        self.num_classes = config.get("num_classes", 1)
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.4)
        self.class_names = config.get("class_names", [f"class_{i}" for i in range(self.num_classes)])
        
        # Note: anchors are now loaded from .npy file instead of config
    
    def _load_anchors(self, anchor_path: str):
        """Load anchors from .npy file"""
        try:
            self.anchors = np.load(anchor_path)
            print(f"✓ Successfully loaded {len(self.anchors)} anchors from {anchor_path}")
            print(f"  Anchor shape: {self.anchors.shape}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Anchor file not found: {anchor_path}")
        except Exception as e:
            print(f"✗ Failed to load anchors: {e}")
            raise
    
    def _load_weights(self, model_path: str):
        """
        Load model weights from .pth or .npz file
        
        Args:
            model_path: Path to model weights
        """
        try:
            if model_path.endswith('.npz'):
                # Load from NumPy format (no torch required)
                numpy_state_dict = {}
                with np.load(model_path, allow_pickle=True) as data:
                    for key in data.files:
                        numpy_state_dict[key] = data[key].item() if data[key].ndim == 0 else data[key]
                self.model.load_weights(numpy_state_dict, self.anchors)
                print(f"✓ Successfully loaded weights from {model_path} (NumPy format)")
            
            elif model_path.endswith('.pth'):
                # Load from PyTorch format (requires torch)
                try:
                    import torch
                except ImportError:
                    raise ImportError(
                        "PyTorch is required to load .pth files. "
                        "Either install PyTorch (pip install torch) or convert weights to .npz format."
                    )
                
                state_dict = torch.load(model_path, map_location='cpu')
                numpy_state_dict = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        numpy_state_dict[key] = value.detach().numpy()
                    else:
                        numpy_state_dict[key] = value
                
                self.model.load_weights(numpy_state_dict, self.anchors)
                print(f"✓ Successfully loaded weights from {model_path} (PyTorch format)")
            
            else:
                raise ValueError(f"Unsupported file format: {model_path}. Use .pth or .npz")
        
        except Exception as e:
            print(f"✗ Weight loading failed: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (H, W, 3) BGR format
        
        Returns:
            Tuple of (processed_image, scale, pad_x, pad_y)
        """
        h, w = image.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Pad to input size
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad image
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=(128, 128, 128))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, 0)  # Add batch dimension
        
        return image, scale, pad_left, pad_top
    
    def postprocess_detections(self, detections: np.ndarray, scale: float, 
                              pad_x: int, pad_y: int) -> List[Dict]:
        """
        Postprocess model outputs to detections
        
        Args:
            detections: Model output (1, C_out, H_out, W_out)
            scale: Scale factor from preprocessing
            pad_x: X padding
            pad_y: Y padding
        
        Returns:
            List of detections [{"bbox": [x1, y1, x2, y2], "confidence": score, "class": class_id}, ...]
        """
        detections_list = []
        
        # Parse detections from output
        # Assuming output format: (batch, num_anchors * (5 + num_classes), grid_h, grid_w)
        batch_size, channels, grid_h, grid_w = detections.shape
        num_anchors = 5
        
        predictions = []
        
        for h in range(grid_h):
            for w in range(grid_w):
                for a in range(num_anchors):
                    offset = a * (5 + self.num_classes)
                    
                    # Parse prediction (raw values)
                    tx = detections[0, offset + 0, h, w]
                    ty = detections[0, offset + 1, h, w]
                    tw = detections[0, offset + 2, h, w]
                    th = detections[0, offset + 3, h, w]
                    conf_raw = detections[0, offset + 4, h, w]
                    
                    # Apply sigmoid to confidence (CRITICAL: must match comvis)
                    conf = 1.0 / (1.0 + np.exp(-conf_raw))
                    
                    # Filter by confidence (after sigmoid)
                    if conf > self.conf_threshold:
                        # Decode bbox in NORMALIZED [0,1] coordinates (match comvis)
                        # XY: grid position + sigmoid(tx/ty), normalized by grid size
                        x_center = (w + 1.0 / (1.0 + np.exp(-tx))) / grid_w
                        y_center = (h + 1.0 / (1.0 + np.exp(-ty))) / grid_h
                        
                        # WH: exp(clamped tw/th) * anchor, normalized by grid size
                        # Clamp to [-4, 4] to prevent extreme values
                        tw_clamped = np.clip(tw, -4, 4)
                        th_clamped = np.clip(th, -4, 4)
                        
                        anchor_w = self.anchors[a][0] / grid_w
                        anchor_h = self.anchors[a][1] / grid_h
                        
                        box_w = np.exp(tw_clamped) * anchor_w
                        box_h = np.exp(th_clamped) * anchor_h
                        
                        # Convert to corner coordinates (still normalized [0,1])
                        x_min = np.clip(x_center - box_w / 2, 0, 1)
                        y_min = np.clip(y_center - box_h / 2, 0, 1)
                        x_max = np.clip(x_center + box_w / 2, 0, 1)
                        y_max = np.clip(y_center + box_h / 2, 0, 1)
                        
                        # Validate box dimensions
                        if x_max > x_min and y_max > y_min:
                            # Get class prediction (sigmoid applied)
                            if self.num_classes > 0:
                                class_probs_raw = detections[0, offset + 5:offset + 5 + self.num_classes, h, w]
                                class_probs = 1.0 / (1.0 + np.exp(-class_probs_raw))
                                class_id = np.argmax(class_probs)
                            else:
                                class_id = 0
                            
                            # Convert normalized [0,1] to pixel coordinates for final output
                            x1_px = x_min * 416
                            y1_px = y_min * 416
                            x2_px = x_max * 416
                            y2_px = y_max * 416
                            
                            # Denormalize to original image coordinates
                            x1 = (x1_px - pad_x) / scale
                            y1 = (y1_px - pad_y) / scale
                            x2 = (x2_px - pad_x) / scale
                            y2 = (y2_px - pad_y) / scale
                            
                            predictions.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": float(conf),  # Use confidence only (no class multiplication)
                                "class": int(class_id),
                                "class_name": self.class_names[class_id]
                            })
        
        # Apply NMS
        if len(predictions) > 0:
            boxes = np.array([p["bbox"] for p in predictions], dtype=np.float32)
            scores = np.array([p["confidence"] for p in predictions], dtype=np.float32)
            
            keep_indices = nms(boxes, scores, self.nms_threshold, use_ciou=False)
            
            detections_list = [predictions[i] for i in keep_indices]
        
        return detections_list
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Perform inference on image
        
        Args:
            image: Input image (H, W, 3) BGR format
        
        Returns:
            List of detections
        """
        import time
        start_time = time.time()
        
        # Preprocess
        processed_image, scale, pad_x, pad_y = self.preprocess_image(image)
        
        # Forward pass
        detections = self.model.forward(processed_image)
        
        # Postprocess
        results = self.postprocess_detections(detections, scale, pad_x, pad_y)
        
        # Update stats
        elapsed = time.time() - start_time
        self.stats["total_inferences"] += 1
        self.stats["total_time"] += elapsed
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_inferences"]
        
        return results
    
    def get_stats(self) -> Dict:
        """Get inference statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_inferences": 0,
            "total_time": 0.0,
            "avg_time": 0.0
        }


def create_engine(model_path: str, config_path: str, anchor_path: str, input_size: int = 416, backend: str = "resnet34") -> SPARKEngine:
    """
    Factory function to create inference engine
    
    Args:
        model_path: Path to trained model weights (.pth file)
        config_path: Path to configuration JSON
        anchor_path: Path to anchor file (.npy format)
        input_size: Input image size (default 416)
        backend: Model backbone (resnet34 only)
    
    Returns:
        SPARKEngine instance
    """
    return SPARKEngine(model_path, config_path, input_size, backend, anchor_path)
