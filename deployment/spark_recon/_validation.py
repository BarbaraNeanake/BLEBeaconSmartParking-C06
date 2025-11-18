"""
Validation Functions
Handles database validation, hogging violation processing, and false positive correction.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


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
