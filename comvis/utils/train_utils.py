"""
Training utilities for SPARK car detection pipeline
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from .model import YOLOv2ResNet, create_model
from .loss import CIoUYOLOLoss
from .data_utils import create_data_loaders


class ModelTrainer:
    """
    Model trainer with improved training pipeline and monitoring
    """
    
    def __init__(self, config, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'loss_components': []
        }
        
        # Get paths
        self.paths = config.get_paths()
        
    def setup(self) -> None:
        """Setup model, loss, optimizer, and data loaders"""
        print("Setting up training pipeline...")
        
        # Create model
        self.model = create_model(self.config, self.device, pretrained=True)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        # Get anchors from dataset
        anchors = self.train_loader.dataset.get_anchors()
        
        # Create loss function
        self.criterion = CIoUYOLOLoss(
            anchors=anchors,
            device=self.device,
            lambda_coord=self.config.lambda_coord,
            lambda_noobj=self.config.lambda_noobj,
            lambda_obj=self.config.lambda_obj
        )
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.num_epochs, 
            eta_min=1e-6
        )
        
        print("✓ Training pipeline setup complete")
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_components = {'coord_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'class_loss': 0}
        num_batches = len(self.train_loader)
        
        for i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss, loss_components = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_components.items():
                if key != 'total_loss':
                    total_components[key] += value
            
            # Print progress
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / (i + 1)
                print(f'  Batch [{i+1}/{num_batches}], Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}')
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_components = {key: value / num_batches for key, value in total_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_components = {'coord_loss': 0, 'obj_loss': 0, 'noobj_loss': 0, 'class_loss': 0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss, loss_components = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                for key, value in loss_components.items():
                    if key != 'total_loss':
                        total_components[key] += value
        
        avg_loss = total_loss / num_batches
        avg_components = {key: value / num_batches for key, value in total_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(self.paths['model_save_dir'], 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.paths['best_model']
            torch.save(self.model.state_dict(), best_path)
            print(f"✓ New best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint['training_history']
            
            print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
            return True
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def plot_training_history(self) -> None:
        """Plot training history"""
        if not self.training_history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.training_history['learning_rate'], 'g-')
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].grid(True)
        
        # Loss components (latest epoch)
        if self.training_history['loss_components']:
            latest_components = self.training_history['loss_components'][-1]
            components = list(latest_components.keys())
            values = list(latest_components.values())
            
            axes[1, 0].bar(components, values)
            axes[1, 0].set_title('Loss Components (Latest Epoch)')
            axes[1, 0].set_ylabel('Loss Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Val loss trend
        if len(self.training_history['val_loss']) > 1:
            axes[1, 1].plot(epochs, self.training_history['val_loss'], 'r-')
            axes[1, 1].axhline(y=self.best_val_loss, color='orange', linestyle='--', label=f'Best: {self.best_val_loss:.4f}')
            axes[1, 1].set_title('Validation Loss Trend')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Val Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.paths['results_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training plots saved: {plot_path}")
    
    def train(self, resume_from_checkpoint: bool = False) -> None:
        """Main training loop"""
        print(f"\n{'='*60}")
        print("Starting SPARK Car Detection Training")
        print(f"{'='*60}")
        
        # Setup if not already done
        if self.model is None:
            self.setup()
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(self.paths['model_save_dir'], 'checkpoint_latest.pth')
            if os.path.exists(checkpoint_path):
                self.load_checkpoint(checkpoint_path)
        
        # Training configuration
        self.config.print_config()
        
        # Freeze backbone initially
        print("\nPhase 1: Training detection head only...")
        self.model.freeze_backbone()
        
        start_epoch = self.current_epoch
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch+1}/{self.config.num_epochs}]")
            
            # Train
            train_loss, train_components = self.train_epoch()
            
            # Validate
            val_loss, val_components = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            self.training_history['loss_components'].append(train_components)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Components - Coord: {train_components['coord_loss']:.4f}, "
                  f"Obj: {train_components['obj_loss']:.4f}, "
                  f"NoObj: {train_components['noobj_loss']:.4f}")
            
            # Early stopping and model saving
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Unfreeze backbone after some epochs
            if epoch == 10 and hasattr(self.model, 'freeze_backbone'):
                print("\nPhase 2: Unfreezing backbone for full model training...")
                self.model.unfreeze_backbone()
                # Reduce learning rate for full model training
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * 0.1
                print(f"  Reduced learning rate to {self.config.learning_rate * 0.1:.6f}")
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved: {self.paths['final_model']}")
        
        # Save final model
        final_model_path = self.paths['final_model']
        torch.save(self.model.state_dict(), final_model_path)
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\n✓ Training pipeline completed successfully!")
        print(f"✓ Best model: {self.paths['best_model']}")
        print(f"✓ Final model: {final_model_path}")
        print(f"✓ Training plots: {os.path.join(self.paths['results_dir'], 'training_history.png')}")


def train_model(config, device: str = 'cpu', resume: bool = False) -> ModelTrainer:
    """
    Convenience function to train a model with given configuration
    
    Args:
        config: Configuration object
        device: Training device
        resume: Whether to resume from checkpoint
        
    Returns:
        Trained ModelTrainer instance
    """
    trainer = ModelTrainer(config, device)
    trainer.train(resume_from_checkpoint=resume)
    return trainer