# test_pipeline.py - Test the improved training pipeline
import os
import torch
import numpy as np
from data import ParsedYOLODataset
from torchvision import transforms

def test_data_loading():
    """Test if data loading works correctly"""
    print("Testing data loading...")
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "parsed_dataset")
    
    try:
        # Test dataset loading
        train_dataset = ParsedYOLODataset(root_dir, "train", transform=train_transform, img_size=416, augment=True)
        val_dataset = ParsedYOLODataset(root_dir, "val", transform=train_transform, img_size=416, augment=False)
        
        print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
        print(f"✅ Val dataset loaded: {len(val_dataset)} samples")
        
        # Test sample loading
        if len(train_dataset) > 0:
            img, targets = train_dataset[0]
            print(f"✅ Sample loaded - Image shape: {img.shape}, Targets shape: {targets.shape}")
            
            # Check for any objects in the sample
            num_objects = (targets[..., 4] > 0).sum().item()
            print(f"✅ Number of objects in first sample: {num_objects}")
            
            # Test anchors
            anchors = train_dataset.get_anchors()
            print(f"✅ Anchors loaded: {anchors.shape} - {anchors}")
            
        else:
            print("❌ No samples found in training dataset")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False
    
    return True

def test_loss_function():
    """Test if loss function works correctly"""
    print("\nTesting loss function...")
    
    try:
        from train import YOLOLoss
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy anchors
        anchors = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        
        # Initialize loss function
        criterion = YOLOLoss(anchors, device, lambda_coord=5.0, lambda_noobj=0.5)
        
        # Create dummy predictions and targets
        batch_size = 2
        num_anchors = 5
        grid_size = 13
        num_classes = 1
        
        # Predictions shape: [batch, num_anchors*(5+classes), grid, grid]
        predictions = torch.randn(batch_size, num_anchors * (5 + num_classes), grid_size, grid_size).to(device)
        
        # Targets shape: [batch, num_anchors, grid, grid, 5]
        targets = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 5).to(device)
        
        # Add a dummy object in the center
        targets[0, 0, 6, 6] = torch.tensor([0.5, 0.5, 0.0, 0.0, 1.0])  # x, y, log_w, log_h, conf
        
        # Test loss computation
        loss = criterion(predictions, targets)
        
        print(f"✅ Loss computed successfully: {loss.item():.4f}")
        
        # Test backward pass
        loss.backward()
        print("✅ Backward pass successful")
        
    except Exception as e:
        print(f"❌ Loss function test failed: {str(e)}")
        return False
    
    return True

def test_anchor_assignment():
    """Test IoU-based anchor assignment"""
    print("\nTesting anchor assignment...")
    
    try:
        root_dir = os.path.join(os.getcwd(), "datasets", "COCO_car", "parsed_dataset")
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ParsedYOLODataset(root_dir, "train", transform=train_transform, img_size=416, augment=False)
        
        if len(dataset) > 0:
            # Test anchor assignment method
            test_box_wh = [2.0, 3.0]  # Example box dimensions
            best_anchor_idx = dataset.find_best_anchor(test_box_wh)
            
            print(f"✅ Best anchor for box {test_box_wh}: anchor {best_anchor_idx}")
            print(f"✅ Anchor dimensions: {dataset.anchors[best_anchor_idx]}")
            
        else:
            print("❌ No dataset samples to test anchor assignment")
            return False
            
    except Exception as e:
        print(f"❌ Anchor assignment test failed: {str(e)}")
        return False
    
    return True

def main():
    print("🔧 Testing Improved YOLO Pipeline\n")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_data_loading():
        tests_passed += 1
    
    if test_loss_function():
        tests_passed += 1
    
    if test_anchor_assignment():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your pipeline is ready for training.")
        print("\n📝 Next steps:")
        print("   1. Re-run datasetPrep.py to regenerate dataset with improvements")
        print("   2. Run train.py to start training with the enhanced pipeline")
        print("   3. Monitor loss curves for better convergence")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()