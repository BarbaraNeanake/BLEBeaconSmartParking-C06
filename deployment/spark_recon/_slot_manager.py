"""
Slot Region Manager
Manages parking slot region definitions and storage.
"""

import logging
import json
from typing import Dict
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
