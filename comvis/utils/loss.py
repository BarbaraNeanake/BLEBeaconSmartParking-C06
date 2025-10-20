"""
Loss functions for SPARK car detection pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) loss for bounding box regression
    
    Args:
        pred_boxes: Predicted boxes [N, 4] in corner format (x1,y1,x2,y2) normalized [0,1]
        target_boxes: Target boxes [N, 4] in corner format (x1,y1,x2,y2) normalized [0,1]
        eps: Small epsilon for numerical stability
        
    Returns:
        CIoU loss sum over N boxes
    """
    x1_p, y1_p, x2_p, y2_p = pred_boxes.unbind(dim=1)
    x1_t, y1_t, x2_t, y2_t = target_boxes.unbind(dim=1)

    # Intersection
    inter_x1 = torch.max(x1_p, x1_t)
    inter_y1 = torch.max(y1_p, y1_t)
    inter_x2 = torch.min(x2_p, x2_t)
    inter_y2 = torch.min(y2_p, y2_t)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area_p = (x2_p - x1_p).clamp(min=0) * (y2_p - y1_p).clamp(min=0)
    area_t = (x2_t - x1_t).clamp(min=0) * (y2_t - y1_t).clamp(min=0)
    union = area_p + area_t - inter_area + eps
    iou = inter_area / union

    # Center distance
    cx_p = (x1_p + x2_p) / 2
    cy_p = (y1_p + y2_p) / 2
    cx_t = (x1_t + x2_t) / 2
    cy_t = (y1_t + y2_t) / 2
    rho2 = (cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2

    # Enclosing box diagonal
    en_x1 = torch.min(x1_p, x1_t)
    en_y1 = torch.min(y1_p, y1_t)
    en_x2 = torch.max(x2_p, x2_t)
    en_y2 = torch.max(y2_p, y2_t)
    en_w = (en_x2 - en_x1).clamp(min=eps)
    en_h = (en_y2 - en_y1).clamp(min=eps)
    c2 = en_w ** 2 + en_h ** 2

    # Aspect ratio term
    w_p = (x2_p - x1_p).clamp(min=eps)
    h_p = (y2_p - y1_p).clamp(min=eps)
    w_t = (x2_t - x1_t).clamp(min=eps)
    h_t = (y2_t - y1_t).clamp(min=eps)
    v = (4 / (torch.pi ** 2)) * (torch.atan(w_t / h_t) - torch.atan(w_p / h_p)) ** 2
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU
    ciou = iou - (rho2 / (c2 + eps)) - alpha * v
    loss = 1 - ciou
    return loss.sum()


class CIoUYOLOLoss(nn.Module):
    """
    Improved YOLO Loss with CIoU for better bounding box regression
    """
    
    def __init__(self, anchors: torch.Tensor, device: str, 
                 lambda_coord: float = 10.0, lambda_noobj: float = 0.5, lambda_obj: float = 1.0):
        super(CIoUYOLOLoss, self).__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32).to(device)
        self.device = device
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def _convert_to_corner_format(self, pred_xy: torch.Tensor, pred_wh: torch.Tensor, 
                                  grid_x: torch.Tensor, grid_y: torch.Tensor, 
                                  anchor_idx: int, grid_size: int) -> torch.Tensor:
        """Convert YOLO format to corner format for CIoU calculation"""
        # Calculate center coordinates
        x_center = (grid_x + pred_xy[..., 0]) / grid_size
        y_center = (grid_y + pred_xy[..., 1]) / grid_size
        
        # Calculate width and height
        anchor_w = self.anchors[anchor_idx, 0] / grid_size
        anchor_h = self.anchors[anchor_idx, 1] / grid_size
        w = torch.exp(pred_wh[..., 0]) * anchor_w
        h = torch.exp(pred_wh[..., 1]) * anchor_h
        
        # Convert to corner format
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of the loss function
        
        Args:
            predictions: Model predictions [batch, num_anchors*(5+classes), grid, grid]
            targets: Ground truth targets [batch, num_anchors, grid, grid, 5]
            
        Returns:
            Total loss and loss components dictionary
        """
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        num_anchors = self.anchors.size(0)
        num_classes = predictions.size(1) // num_anchors - 5

        # Reshape predictions
        predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        # Apply transformations
        pred_xy = torch.sigmoid(predictions[..., :2])
        pred_wh = predictions[..., 2:4]
        pred_conf = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:]) if num_classes > 0 else None

        # Ensure targets have correct shape
        if targets.dim() == 4:
            targets = targets.view(batch_size, num_anchors, grid_size, grid_size, 5)

        # Object masks
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask

        # Initialize loss components
        coord_loss = torch.tensor(0.0, device=self.device)
        obj_loss = torch.tensor(0.0, device=self.device)
        noobj_loss = torch.tensor(0.0, device=self.device)
        class_loss = torch.tensor(0.0, device=self.device)

        # Calculate coordinate loss using CIoU for objects
        if obj_mask.sum() > 0:
            # Create grid coordinates
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
            grid_x = grid_x.float().to(self.device)
            grid_y = grid_y.float().to(self.device)
            
            total_ciou_loss = 0
            num_objects = 0
            
            for b in range(batch_size):
                for k in range(num_anchors):
                    mask = obj_mask[b, k]
                    if mask.sum() > 0:
                        # Get predictions for objects
                        pred_xy_obj = pred_xy[b, k][mask]
                        pred_wh_obj = pred_wh[b, k][mask]
                        target_xy_obj = targets[b, k, ..., :2][mask]
                        target_wh_obj = targets[b, k, ..., 2:4][mask]
                        
                        # Get grid coordinates for objects
                        grid_x_obj = grid_x[mask]
                        grid_y_obj = grid_y[mask]
                        
                        # Convert predictions to corner format
                        pred_corners = self._convert_to_corner_format(
                            pred_xy_obj, pred_wh_obj,
                            grid_x_obj, grid_y_obj, k, grid_size
                        )
                        
                        # Convert targets to corner format
                        target_x_center = (grid_x_obj + target_xy_obj[:, 0]) / grid_size
                        target_y_center = (grid_y_obj + target_xy_obj[:, 1]) / grid_size
                        target_w = torch.exp(target_wh_obj[:, 0]) * self.anchors[k, 0] / grid_size
                        target_h = torch.exp(target_wh_obj[:, 1]) * self.anchors[k, 1] / grid_size
                        
                        target_x1 = target_x_center - target_w / 2
                        target_y1 = target_y_center - target_h / 2
                        target_x2 = target_x_center + target_w / 2
                        target_y2 = target_y_center + target_h / 2
                        target_corners = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)
                        
                        # Clamp to valid ranges
                        pred_corners = torch.clamp(pred_corners, 0, 1)
                        target_corners = torch.clamp(target_corners, 0, 1)
                        
                        # Calculate CIoU loss
                        if pred_corners.size(0) > 0:
                            ciou_loss_val = ciou_loss(pred_corners, target_corners)
                            total_ciou_loss += ciou_loss_val
                            num_objects += pred_corners.size(0)
            
            if num_objects > 0:
                coord_loss = total_ciou_loss / num_objects

        # Confidence loss
        obj_pred = pred_conf[obj_mask]
        obj_target = targets[..., 4][obj_mask]
        noobj_pred = pred_conf[noobj_mask]
        noobj_target = torch.zeros_like(noobj_pred)

        if obj_pred.numel() > 0:
            obj_loss = self.bce_loss(obj_pred, obj_target).sum() / obj_pred.numel()
        if noobj_pred.numel() > 0:
            noobj_loss = self.bce_loss(noobj_pred, noobj_target).sum() / noobj_pred.numel()

        # Class loss
        if num_classes > 0 and obj_mask.sum() > 0 and pred_cls is not None:
            class_pred = pred_cls[obj_mask]
            class_target = torch.ones_like(class_pred)
            if class_pred.numel() > 0:
                class_loss = self.bce_loss(class_pred, class_target).sum() / class_pred.numel()

        # Combine losses
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            class_loss
        )

        # Loss components for monitoring
        loss_components = {
            'total_loss': total_loss.item(),
            'coord_loss': coord_loss.item() if torch.is_tensor(coord_loss) else coord_loss,
            'obj_loss': obj_loss.item(),
            'noobj_loss': noobj_loss.item(),
            'class_loss': class_loss.item() if torch.is_tensor(class_loss) else class_loss
        }

        return total_loss, loss_components