#%% Setup and Imports
import os
import sys
import numpy as np
from typing import Optional
import torch
from PIL import Image

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import utils modules
from utils.config import Config, default_config
from utils.train_utils import train_model
from utils.inference_utils import run_inference
from utils.data_utils import compute_anchors


#%% Device Setup
def setup_device() -> str:
    """Setup and return the best available device"""
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = 'cpu'
            print("✓ Using CPU (CUDA not available)")
    except ImportError:
        device = 'cpu'
        print("✓ Using CPU (PyTorch not installed)")
    
    return device

#%% Configuration Management
def load_config(config_path: Optional[str] = None):
    """Load configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        print(f"✓ Loading configuration from: {config_path}")
        return Config.from_json(config_path)
    else:
        # No config path provided - use in-code default configuration
        print("✓ Using built-in high precision default configuration")
        return default_config

def save_config(config, config_path: str) -> None:
    """Save configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config.to_json(config_path)
    print(f"✓ Configuration saved to: {config_path}")

#%% Anchor Preparation
def prepare_anchors(config, force_recompute: bool = False) -> None:
    """Prepare anchor boxes for training"""
    paths = config.get_paths()
    anchors_file = paths['anchors_file']
    
    if os.path.exists(anchors_file) and not force_recompute:
        print(f"✓ Anchors already exist: {anchors_file}")
        return
    
    print("Computing improved anchors...")
    
    # Try to compute anchors from actual dataset
    parsed_dataset = paths['parsed_dataset']
    train_images_dir = os.path.join(parsed_dataset, "train", "images")
    train_labels_dir = os.path.join(parsed_dataset, "train", "labels")
    
    # Check if dataset exists and has samples
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        
        if len(image_files) > 0:
            print(f"✓ Found {len(image_files)} training images")
            print("  Computing anchors from dataset samples...")
            
            try:
                # Load samples for anchor computation
                from PIL import Image
                car_samples = []
                
                # Process up to 500 samples for anchor computation (to avoid memory issues)
                sample_files = image_files[:min(500, len(image_files))]
                
                for img_file in sample_files:
                    img_path = os.path.join(train_images_dir, img_file)
                    label_path = os.path.join(train_labels_dir, img_file.replace('.jpg', '.txt'))
                    
                    if os.path.exists(label_path):
                        try:
                            img = Image.open(img_path)
                            img_width, img_height = img.size
                            
                            # Parse labels
                            annotations = []
                            with open(label_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        cls, x_center, y_center, width, height = map(float, parts)
                                        
                                        # Convert to absolute coordinates for anchor computation
                                        abs_width = width * img_width
                                        abs_height = height * img_height
                                        abs_x = (x_center - width/2) * img_width
                                        abs_y = (y_center - height/2) * img_height
                                        
                                        annotations.append({
                                            'bbox': [abs_x, abs_y, abs_width, abs_height],
                                            'category_id': int(cls)
                                        })
                            
                            if annotations:
                                car_samples.append((img, annotations))
                                
                        except Exception as e:
                            print(f"  Warning: Could not process {img_file}: {e}")
                            continue
                
                if len(car_samples) >= 10:  # Need minimum samples for clustering
                    print(f"  Using {len(car_samples)} samples for anchor computation...")
                    anchors = compute_anchors(car_samples, config, config.num_anchors)
                    
                    os.makedirs(os.path.dirname(anchors_file), exist_ok=True)
                    np.save(anchors_file, anchors)
                    print(f"✓ Computed anchors saved to: {anchors_file}")
                    print(f"  Anchors: {anchors}")
                    return
                    
                else:
                    print(f"  Warning: Only found {len(car_samples)} valid samples, need at least 10")
                    
            except Exception as e:
                print(f"  Warning: Anchor computation failed: {e}")
                print("  Falling back to default anchors...")
    
    # Fallback to default anchors
    print("⚠ Using fallback anchors")
    print("  To get better anchors, ensure your dataset is prepared with:")
    print("  - datasets/COCO_car/parsed_dataset/train/images/")
    print("  - datasets/COCO_car/parsed_dataset/train/labels/")

    fallback_anchors = np.array([
        [1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
        [9.47112, 4.84053], [11.2364, 10.0071]
    ])
    
    os.makedirs(os.path.dirname(anchors_file), exist_ok=True)
    np.save(anchors_file, fallback_anchors)
    print(f"✓ Fallback anchors saved to: {anchors_file}")


#%% Dataset Validation
def validate_dataset(config) -> bool:
    """Validate that dataset exists and is properly formatted"""
    paths = config.get_paths()
    parsed_dataset = paths['parsed_dataset']
    
    required_dirs = [
        os.path.join(parsed_dataset, "train", "images"),
        os.path.join(parsed_dataset, "train", "labels"),
        os.path.join(parsed_dataset, "val", "images"),
        os.path.join(parsed_dataset, "val", "labels")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"✗ Missing required directory: {dir_path}")
            return False
        
        # Check if directory has files
        files = os.listdir(dir_path)
        if len(files) == 0:
            print(f"✗ Empty directory: {dir_path}")
            return False
    
    print(f"✓ Dataset validation passed: {parsed_dataset}")
    return True


#%% Training Pipeline
def train_pipeline(config=None, device=None, resume: bool = False) -> None:
    """Run the training pipeline"""
    
    if config is None:
        config = load_config()
    if device is None:
        device = setup_device()
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60)
    
    # Validate dataset
    if not validate_dataset(config):
        print("✗ Dataset validation failed. Please prepare your dataset first.")
        print("  Run your datasetPrep.py script to create the parsed dataset.")
        return
    
    # Prepare anchors
    prepare_anchors(config)
    
    # Start training
    try:
        trainer = train_model(config, device, resume)
        print("✓ Training pipeline completed successfully!")
        
        # Save training configuration
        config_path = os.path.join(config.get_paths()['model_save_dir'], 'training_config.json')
        save_config(config, config_path)
        
    except Exception as e:
        print(f"✗ Training failed: {str(e)}")
        raise


#%% Inference Pipeline
def inference_pipeline(config=None, device=None, model_path: Optional[str] = None,
                      test_images_dir: Optional[str] = None) -> None:
    """Run the inference pipeline"""
    
    if config is None:
        config = load_config()
    if device is None:
        device = setup_device()
    
    print("\n" + "="*60)
    print("INFERENCE PIPELINE")
    print("="*60)
    
    try:
        inference = run_inference(config, model_path, test_images_dir, device)
        print("✓ Inference pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Inference failed: {str(e)}")
        raise


#%% Full Pipeline
def full_pipeline(config=None, device=None) -> None:
    """Run the complete pipeline (train + inference)"""
    
    if config is None:
        config = load_config()
    if device is None:
        device = setup_device()
    
    print("\n" + "="*60)
    print("FULL PIPELINE")
    print("="*60)
    
    # Train
    train_pipeline(config, device, resume=False)
    
    # Inference
    inference_pipeline(config, device)
    
    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETED!")
    print("="*60)


#%% Compute Anchors
def compute_anchors(config=None, force_recompute: bool = True) -> None:
    """Compute and save anchor boxes"""
    
    if config is None:
        config = load_config()
    
    print("\n" + "="*60)
    print("ANCHOR COMPUTATION")
    print("="*60)
    
    prepare_anchors(config, force_recompute)


#%% Show Configuration
def show_config(config=None) -> None:
    """Display current configuration"""
    
    if config is None:
        config = load_config()
    
    config.print_config()


#%% Quick Start - Setup Device and Config
# Run this cell first to setup your environment
device = setup_device()
config = load_config()
show_config(config)

#%% Train Model
# Run this cell to train your model
train_pipeline(config, device, resume=False)

#%% Resume Training
# Run this cell to resume training from checkpoint
# train_pipeline(config, device, resume=True)

#%% Run Inference
# Run this cell to test your trained model
inference_pipeline(config, device)

#%% Run Full Pipeline
# Run this cell to train and then test your model
# full_pipeline(config, device)

#%% Compute Custom Anchors
# Run this cell to recompute anchors from your dataset
# compute_anchors(config, force_recompute=True)