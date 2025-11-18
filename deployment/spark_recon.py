"""
DEPRECATED: This file has been replaced by the spark_recon module (folder).

The SPARK Reconnaissance Module has been refactored into separate files:
- spark_recon/_slot_manager.py - Slot region management
- spark_recon/_hogging.py - Hogging detection algorithm
- spark_recon/_preprocessing.py - Image preprocessing
- spark_recon/_validation.py - Database validation

Use: from spark_recon import SlotRegionManager, hogging_detection, etc.

This file can be safely deleted after verifying the new structure works.
"""

# For backward compatibility, re-export from the new module structure
from spark_recon import (
    SlotRegionManager,
    hogging_detection,
    preprocess_uploaded_image,
    serialize_detections,
    fetch_relevant_slots,
    handle_hogging_violations,
    correct_false_positives
)

__all__ = [
    'SlotRegionManager',
    'hogging_detection',
    'preprocess_uploaded_image',
    'serialize_detections',
    'fetch_relevant_slots',
    'handle_hogging_violations',
    'correct_false_positives'
]


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
        "total_slots": len(slot_regions_dict)
    }

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


async def fetch_relevant_slots(db_manager, slot_regions: Dict) -> Tuple[Optional[Dict], Optional[Dict], Optional[List[str]]]:
    """
    Stage 1: Fetch occupied slots and neighbors from database.
    
    Args:
        db_manager: DatabaseManager instance
        slot_regions: Dict of all slot regions
    
    Returns:
        Tuple of (slot_data, relevant_slot_regions, occupied_slots)
        Returns (None, None, None) if no occupied slots found
    """
    slot_data = await db_manager.get_occupied_slots_with_neighbors()
    
    if not slot_data:
        logger.info("ℹ️ No occupied slots found, skipping hogging detection")
        return None, None, None
    
    occupied_count = sum(1 for s in slot_data.values() if s["query_reason"] == "occupied")
    neighbor_count = sum(1 for s in slot_data.values() if s["query_reason"] == "neighbor")
    logger.info(f"📊 Stage 1: Retrieved {occupied_count} occupied + {neighbor_count} neighbor slots")
    
    # Filter to relevant regions
    relevant_slot_regions = {
        slot_id: slot_regions[slot_id] 
        for slot_id in slot_data.keys() 
        if slot_id in slot_regions
    }
    
    occupied_slots = [slot_id for slot_id, data in slot_data.items() if data["status"] == 'occupied']
    
    return slot_data, relevant_slot_regions, occupied_slots


async def handle_hogging_violations(violations: List[Dict], slot_data: Dict, db_manager):
    """
    Case 2: Process hogging violations and log to database.
    
    Args:
        violations: List of hogging violation dicts
        slot_data: Dict of slot data with userid info
        db_manager: DatabaseManager instance
    """
    if not violations:
        return
    
    logger.warning(f"⚠️ Detected {len(violations)} slot hogging violations!")
    
    for violation in violations:
        slots_str = ", ".join(violation['occupied_slots'])
        logger.warning(f"   🚗 Car (conf: {violation['confidence']:.2f}) hogging slots: {slots_str}")
        
        # Find userid from occupied slots (skip userid=0)
        violator_userid = None
        violator_slot = None
        
        for slot_id in violation['occupied_slots']:
            slot_info = slot_data.get(slot_id, {})
            userid = slot_info.get('userid', 0)
            if userid > 0:
                violator_userid = userid
                violator_slot = slot_id
                break
        
        # Insert violation
        if violator_userid and violator_slot:
            inserted = await db_manager.insert_pelanggaran(
                userid=violator_userid,
                nomor=violator_slot,
                jenis_pelanggaran="Slot Hogging"
            )
            if inserted:
                logger.info(f"📝 Logged NEW violation for userid={violator_userid} at slot {violator_slot}")
            else:
                logger.info(f"⏭️ Violation already exists for userid={violator_userid} at slot {violator_slot}")
        else:
            logger.warning(f"⚠️ No valid userid found in hogging slots: {slots_str}")


async def correct_false_positives(occupied_slots: List[str], slot_occupancy: Dict, db_manager):
    """
    Case 3: Correct slots marked as occupied but with no car detected.
    
    Args:
        occupied_slots: List of slot IDs that DB says are occupied
        slot_occupancy: Dict from hogging analysis with camera detection results
        db_manager: DatabaseManager instance
    """
    for slot_id in occupied_slots:
        slot_camera_status = slot_occupancy.get(slot_id, {})
        is_car_detected = slot_camera_status.get('is_occupied', False)
        
        # DB says occupied but camera doesn't detect a car
        if not is_car_detected:
            logger.warning(f"⚠️ Slot {slot_id}: DB says 'occupied' but no car detected")
            logger.info(f"🔄 Updating slot {slot_id} status to 'available'")
            
            await db_manager.update_slot_status(slot_id, False)
            logger.info(f"✅ Slot {slot_id} status corrected")
