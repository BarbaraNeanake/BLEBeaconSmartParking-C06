#!/usr/bin/env python3
import cv2
import sys
import torch
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = "../comvis/models/best_model.pth"
CONFIG_PATH = "config.json"
TEST_IMAGE = "../comvis/test_images/single-car-empty-parking-lot-549.jpg"

def test_engine():
    print("=" * 50)
    print("SPARK Engine Local Test (PyTorch)")
    print("=" * 50)
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    if not Path(CONFIG_PATH).exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    
    if not Path(TEST_IMAGE).exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    # Load PyTorch model
    print("\n🔧 Loading PyTorch model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print(f"✅ Model loaded")
        print(f"   Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Direct state dict'}")
        
        # Get state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"   Number of parameters: {len(state_dict)}")
        print(f"   Sample parameters: {list(state_dict.keys())[:5]}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load image
    print(f"\n📷 Loading test image: {TEST_IMAGE}")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"❌ Failed to load image")
        return
    
    print(f"   Image shape: {image.shape}")
    print(f"   Image dtype: {image.dtype}")
    
    print("\n✅ Test complete - Model loaded successfully")

if __name__ == "__main__":
    test_engine()
