"""
Image Preprocessing
Handles image preprocessing and detection serialization.
"""

import io
from typing import Dict, List
import numpy as np
import cv2
from PIL import Image


def preprocess_uploaded_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert uploaded image to BGR format for model inference.
    
    Args:
        image_bytes: Raw image bytes from upload
    
    Returns:
        numpy array in BGR format (H, W, 3)
    """
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_array = np.array(image, dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def serialize_detections(detections: List[Dict]) -> List[Dict]:
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        detections: List of detection dicts with NumPy types
    
    Returns:
        List of detections with native Python types
    """
    return [
        {
            "bbox": [float(x) for x in det.get('bbox', [])],
            "confidence": float(det.get('confidence', 0)),
            "class": int(det.get('class', 0)),
            "class_name": str(det.get('class_name', 'unknown'))
        }
        for det in detections
    ]
