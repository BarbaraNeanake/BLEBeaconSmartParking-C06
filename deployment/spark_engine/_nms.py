"""
Pure Python NMS (Non-Maximum Suppression) implementation using NumPy
Supports both IoU and CIoU metrics
"""

import numpy as np
from typing import Tuple, List


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes
    
    Args:
        box_a: Box coordinates [x1, y1, x2, y2]
        box_b: Box coordinates [x1, y1, x2, y2]
    
    Returns:
        IoU value (0-1)
    """
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    union_area = box_a_area + box_b_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_ciou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Complete IoU (CIoU) between two boxes
    
    Args:
        box_a: Box coordinates [x1, y1, x2, y2]
        box_b: Box coordinates [x1, y1, x2, y2]
    
    Returns:
        CIoU value
    """
    # Basic IoU
    x1_inter = max(box_a[0], box_b[0])
    y1_inter = max(box_a[1], box_b[1])
    x2_inter = min(box_a[2], box_b[2])
    y2_inter = min(box_a[3], box_b[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    
    # Distance penalty
    c_x1 = min(box_a[0], box_b[0])
    c_y1 = min(box_a[1], box_b[1])
    c_x2 = max(box_a[2], box_b[2])
    c_y2 = max(box_a[3], box_b[3])
    
    c_diag_squared = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
    
    box_a_cx = (box_a[0] + box_a[2]) / 2
    box_a_cy = (box_a[1] + box_a[3]) / 2
    box_b_cx = (box_b[0] + box_b[2]) / 2
    box_b_cy = (box_b[1] + box_b[3]) / 2
    
    d_squared = (box_a_cx - box_b_cx) ** 2 + (box_a_cy - box_b_cy) ** 2
    
    distance_penalty = d_squared / c_diag_squared if c_diag_squared > 0 else 0.0
    
    # Aspect ratio consistency
    w_a = box_a[2] - box_a[0]
    h_a = box_a[3] - box_a[1]
    w_b = box_b[2] - box_b[0]
    h_b = box_b[3] - box_b[1]
    
    v = 0.0
    if w_a > 0 and h_a > 0 and w_b > 0 and h_b > 0:
        atan_a = w_a / h_a
        atan_b = w_b / h_b
        v = 4.0 / (np.pi ** 2) * (atan_a - atan_b) ** 2
    
    alpha = v / (1 - iou + v) if (1 - iou + v) > 0 else 0.0
    
    ciou = iou - distance_penalty - alpha * v
    return ciou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5,
        use_ciou: bool = False) -> np.ndarray:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Box coordinates (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        use_ciou: Use CIoU instead of IoU
    
    Returns:
        Indices of kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        compute_fn = compute_ciou if use_ciou else compute_iou
        ious = np.array([compute_fn(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes with IoU below threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5,
             sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression (weighted NMS)
    
    Args:
        boxes: Box coordinates (N, 4) in [x1, y1, x2, y2] format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold
        sigma: Sigma for Gaussian weighting
    
    Returns:
        Tuple of (kept_indices, adjusted_scores)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    
    # Sort by score
    order = np.argsort(scores)[::-1]
    scores = scores.copy()
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append((i, scores[i]))
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Penalize scores
        weights = np.exp(-(ious ** 2) / sigma)
        scores[order[1:]] *= weights
        
        # Keep boxes above threshold
        inds = np.where(scores[order[1:]] > 0.01)[0]
        order = order[inds + 1]
    
    keep_indices = np.array([k[0] for k in keep], dtype=np.int32)
    keep_scores = np.array([k[1] for k in keep], dtype=np.float32)
    
    return keep_indices, keep_scores
