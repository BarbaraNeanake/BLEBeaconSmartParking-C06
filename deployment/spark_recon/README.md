# SPARK Reconnaissance Module

This module handles parking slot region management and violation detection (hogging).

## Structure

- `_slot_manager.py` - Slot region management and JSON storage
- `_hogging.py` - Hogging detection algorithm (X-axis bounds method)
- `_preprocessing.py` - Image preprocessing and detection serialization
- `_validation.py` - Database validation and false positive correction

## Usage

```python
from spark_recon import (
    SlotRegionManager,
    hogging_detection,
    preprocess_uploaded_image,
    serialize_detections,
    fetch_relevant_slots,
    handle_hogging_violations,
    correct_false_positives
)
```

## Components

### SlotRegionManager
Manages parking slot region definitions with load/save functionality.

### hogging_detection()
Detects cars occupying multiple parking slots using X-axis overlap detection.

### preprocess_uploaded_image()
Converts uploaded image bytes to BGR format for model inference.

### serialize_detections()
Converts NumPy detection results to JSON-serializable format.

### fetch_relevant_slots()
Fetches occupied slots and neighbors from database (Stage 1).

### handle_hogging_violations()
Processes hogging violations and logs to database (Case 2).

### correct_false_positives()
Corrects false positive occupancy status (Case 3).
