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
from utils import Config, default_config, train_model, run_inference, compute_anchors


#%% Device Setup
def setup_device() -> str:
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

#%% Configuration Management
def load_config(config_path: Optional[str] = None):
    """Load configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        return Config.from_json(config_path)
    else:
        return default_config

def save_config(config, config_path: str) -> None:
    """Save configuration to file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config.to_json(config_path)

#%% Anchor Preparation
def prepare_anchors(config, force_recompute: bool = False) -> None:
    """Prepare anchor boxes for training"""
    paths = config.get_paths()
    anchors_file = paths['anchors_file']
    
    if os.path.exists(anchors_file) and not force_recompute:
        return
    
    print("Computing anchors from dataset...")
    
    # Try to compute anchors from actual dataset
    parsed_dataset = paths['parsed_dataset']
    train_images_dir = os.path.join(parsed_dataset, "train", "images")
    train_labels_dir = os.path.join(parsed_dataset, "train", "labels")
    
    # Check if dataset exists
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        
        if len(image_files) > 0:
            try:
                car_samples = []
                sample_files = image_files[:min(500, len(image_files))]
                
                for img_file in sample_files:
                    img_path = os.path.join(train_images_dir, img_file)
                    label_path = os.path.join(train_labels_dir, img_file.replace('.jpg', '.txt'))
                    
                    if os.path.exists(label_path):
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        
                        annotations = []
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    cls, x_center, y_center, width, height = map(float, parts)
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
                
                if len(car_samples) >= 10:
                    anchors = compute_anchors(car_samples, config, config.num_anchors)
                    os.makedirs(os.path.dirname(anchors_file), exist_ok=True)
                    np.save(anchors_file, anchors)
                    print(f"Anchors computed: {anchors}")
                    return
                    
            except Exception as e:
                print(f"Anchor computation failed: {e}")
    
    # Fallback to default anchors
    print("Using default anchors")
    fallback_anchors = np.array([
        [1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], 
        [9.47112, 4.84053], [11.2364, 10.0071]
    ])
    
    os.makedirs(os.path.dirname(anchors_file), exist_ok=True)
    np.save(anchors_file, fallback_anchors)


#%% Dataset Validation
def validate_dataset(config) -> bool:
    """Validate dataset exists"""
    paths = config.get_paths()
    parsed_dataset = paths['parsed_dataset']
    
    required_dirs = [
        os.path.join(parsed_dataset, "train", "images"),
        os.path.join(parsed_dataset, "train", "labels"),
        os.path.join(parsed_dataset, "val", "images"),
        os.path.join(parsed_dataset, "val", "labels")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path) or len(os.listdir(dir_path)) == 0:
            print(f"Missing or empty: {dir_path}")
            return False
    
    return True


#%% Training Pipeline
def train_pipeline(config=None, device=None, resume: bool = False) -> None:
    """Train the model"""
    
    if config is None:
        config = load_config()
    if device is None:
        device = setup_device()
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    if not validate_dataset(config):
        print("Dataset validation failed. Run datasetPrep.py first.")
        return
    
    prepare_anchors(config)
    
    try:
        train_model(config, device, resume)
        print("Training completed")
        
        # Save config
        config_path = os.path.join(config.get_paths()['model_save_dir'], 'training_config.json')
        save_config(config, config_path)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise


#%% Inference Pipeline
def inference_pipeline(config=None, device=None, model_path: Optional[str] = None,
                      test_images_dir: Optional[str] = None) -> None:
    """Test the model"""
    
    if config is None:
        config = load_config()
    if device is None:
        device = setup_device()
    
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    
    try:
        run_inference(config, model_path, test_images_dir, device)
        print("Testing completed")
        
    except Exception as e:
        print(f"Testing failed: {str(e)}")
        raise


#%% Compute Anchors
def compute_anchors_pipeline(config=None, force_recompute: bool = True) -> None:
    """Compute anchor boxes"""
    
    if config is None:
        config = load_config()
    prepare_anchors(config, force_recompute)


#%% Quick Start
device = setup_device()
config = load_config()
config.print_config()

#%% Train Model
train_pipeline(config, device)

#%% Test Model
inference_pipeline(config, device)

#%% Compute Anchors (Optional)
compute_anchors_pipeline(config, force_recompute=True)
