#!/usr/bin/env python3
import cv2
import sys
import requests
import numpy as np
from pathlib import Path

# Configuration
API_URL = "https://danishritonga-spark-backend.hf.space/detect"
TEST_IMAGE = "single-car-empty-parking-lot-549.jpg"

def test_engine():
    print("=" * 50)
    print("SPARK API Endpoint Test")
    print("=" * 50)
    
    # Check if test image exists
    if not Path(TEST_IMAGE).exists():
        print(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    # Load image
    print(f"\n� Loading test image: {TEST_IMAGE}")
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"❌ Failed to load image")
        return
    
    print(f"   Image shape: {image.shape}")
    print(f"   Image size: {image.shape[0]}x{image.shape[1]}")
    
    # Encode image to send via API
    print("\n� Encoding image...")
    _, img_encoded = cv2.imencode('.jpg', image)
    
    # Send request to API
    print(f"\n🚀 Sending request to: {API_URL}")
    try:
        files = {'file': ('test_image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful")
            print(f"\n📊 API Response:")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Inference time: {result.get('inference_time', 0):.4f}s")
            print(f"   Image shape: {result.get('image_shape', [])}")
            print(f"   Detections: {result.get('num_detections', 0)}")
            
            # Show detections
            detections = result.get('detections', [])
            if detections:
                print("\n🎯 Detection Results:")
                for i, det in enumerate(detections, 1):
                    bbox = det.get('bbox', [])
                    confidence = det.get('confidence', 0)
                    class_name = det.get('class_name', 'N/A')
                    print(f"   {i}. Class: {class_name}")
                    print(f"      Confidence: {confidence:.4f}")
                    print(f"      Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            else:
                print("\n⚠️  No detections found")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - Is the API running at {API_URL}?")
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
    except Exception as e:
        print(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_engine()
