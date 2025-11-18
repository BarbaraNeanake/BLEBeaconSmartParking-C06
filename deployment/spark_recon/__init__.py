"""
SPARK Reconnaissance Module
Handles parking slot region management and violation detection (hogging).
"""

from ._slot_manager import SlotRegionManager
from ._hogging import hogging_detection
from ._preprocessing import preprocess_uploaded_image, serialize_detections
from ._validation import (
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
