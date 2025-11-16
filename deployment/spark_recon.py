"""
SPARK Reconnaissance Module
Handles parking slot region management and violation detection (hogging).
"""

import logging
import json
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SlotRegionManager:
    """Manages parking slot region definitions and storage."""
    
    def __init__(self):
        self.regions: Dict[str, Dict] = {}
    
    def load(self, slot_regions_file: str):
        """
        Load parking slot region definitions from JSON file.
        
        Expected format:
        {
            "A01": {
                "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "name": "Slot A01"
            },
            "A02": { ... }
        }
        
        Args:
            slot_regions_file: Path to the JSON file containing slot regions
        
        Returns:
            Self for method chaining
        """
        try:
            file_path = Path(slot_regions_file)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.regions = json.load(f)
                logger.info(f"✅ Loaded {len(self.regions)} slot regions from {file_path}")
            else:
                logger.warning(f"⚠️ Slot regions file not found: {file_path}")
                self.regions = {}
        except Exception as e:
            logger.error(f"❌ Failed to load slot regions: {e}")
            self.regions = {}
        
        return self
    
    def get(self) -> Dict[str, Dict]:
        """
        Get the currently loaded slot regions.
        
        Returns:
            Dict of slot regions
        """
        return self.regions
    
    def save(self, regions: Dict[str, Dict], slot_regions_file: str):
        """
        Save slot regions to file and update in-memory storage.
        
        Args:
            regions: Dict of slot regions to save
            slot_regions_file: Path to save the JSON file
        """
        try:
            file_path = Path(slot_regions_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(regions, f, indent=2)
            
            # Update in-memory
            self.regions = regions
            
            logger.info(f"✅ Saved {len(regions)} slot regions to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save slot regions: {e}")
            raise


def hogging_detection(detections: List[Dict], slot_regions_dict: Dict[str, Dict]) -> Dict:
    """
    Detect if any car is hogging multiple parking slots using X-axis bounds.
    Simple method: checks if car's x_min and x_max fall outside any single slot's x bounds.
    
    Args:
        detections: List of detection dicts with 'bbox' key [x_min, y_min, x_max, y_max]
        slot_regions_dict: Dict of {slot_id: {corners: [[x,y], ...], name: ...}}
    
    Returns:
        Dict with hogging analysis:
        {
            "violations": [...],
            "slot_occupancy": {...},
            "total_violations": int,
            "method": "x_axis_bounds"
        }
    
    Note: This is a simple method with limitations:
    - Ignores Y-axis (depth/row information)
    - May give false positives for multi-row parking
    - Best for single-row, side-by-side parking slots
    """
    violations = []
    slot_occupancy = {}
    
    # Initialize slot occupancy
    for slot_id in slot_regions_dict.keys():
        slot_occupancy[slot_id] = {
            "is_occupied": False,
            "car_count": 0,
            "cars": [],
            "is_violation": False
        }
    
    # Calculate X-bounds for each slot
    slot_x_bounds = {}
    for slot_id, slot_data in slot_regions_dict.items():
        corners = slot_data.get('corners', [])
        if len(corners) < 3:
            logger.warning(f"⚠️ Slot {slot_id}: Invalid corners (need at least 3)")
            continue
        
        x_coords = [c[0] for c in corners]
        slot_x_bounds[slot_id] = {
            "x_min": min(x_coords),
            "x_max": max(x_coords)
        }
    
    # Check each detected car
    for detection in detections:
        car_bbox = detection.get('bbox', [])
        if len(car_bbox) != 4:
            continue
        
        x_min_car, y_min_car, x_max_car, y_max_car = car_bbox
        occupied_slots = []
        slot_overlaps = {}
        
        # Check which slots this car overlaps with (X-axis only)
        for slot_id, bounds in slot_x_bounds.items():
            x_min_slot = bounds["x_min"]
            x_max_slot = bounds["x_max"]
            
            # Check if car's X-range overlaps with slot's X-range
            # Overlap exists if: NOT (car completely left of slot OR car completely right of slot)
            x_overlap = not (x_max_car < x_min_slot or x_min_car > x_max_slot)
            
            if x_overlap:
                # Calculate overlap amount (for debugging/analysis)
                overlap_start = max(x_min_car, x_min_slot)
                overlap_end = min(x_max_car, x_max_slot)
                overlap_width = overlap_end - overlap_start
                car_width = x_max_car - x_min_car
                overlap_percentage = overlap_width / car_width if car_width > 0 else 0
                
                occupied_slots.append(slot_id)
                slot_overlaps[slot_id] = {
                    "overlap_percentage": float(overlap_percentage),
                    "car_x_range": [float(x_min_car), float(x_max_car)],
                    "slot_x_range": [float(x_min_slot), float(x_max_slot)]
                }
                
                # Update slot occupancy
                slot_occupancy[slot_id]["is_occupied"] = True
                slot_occupancy[slot_id]["car_count"] += 1
                slot_occupancy[slot_id]["cars"].append({
                    "bbox": car_bbox,
                    "confidence": float(detection.get('confidence', 0)),
                    "overlap_percentage": float(overlap_percentage)
                })
        
        # If car occupies multiple slots, it's a violation (hogging)
        if len(occupied_slots) > 1:
            violation = {
                "car_bbox": car_bbox,
                "confidence": float(detection.get('confidence', 0)),
                "occupied_slots": occupied_slots,
                "num_slots": len(occupied_slots),
                "slot_overlaps": slot_overlaps,
                "violation_type": "slot_hogging",
                "severity": "high" if len(occupied_slots) >= 3 else "medium"
            }
            violations.append(violation)
            
            # Mark all affected slots as violations
            for slot_id in occupied_slots:
                slot_occupancy[slot_id]["is_violation"] = True
    
    return {
        "violations": violations,
        "slot_occupancy": slot_occupancy,
        "total_violations": len(violations),
        "affected_slots": [slot_id for slot_id, data in slot_occupancy.items() if data["is_violation"]],
        "total_slots": len(slot_regions_dict),
        "method": "x_axis_bounds",
        "note": "Simple X-axis method. Limitations: ignores Y-axis, may have false positives in multi-row parking."
    }
