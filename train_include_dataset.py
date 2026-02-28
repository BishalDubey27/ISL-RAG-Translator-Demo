#!/usr/bin/env python3
"""
INCLUDE Dataset Training Pipeline
Complete training system for ISL sign recognition using the INCLUDE dataset

The INCLUDE dataset contains:
- 263 ISL signs
- Multiple videos per sign
- Diverse signers and conditions
- High-quality annotations

Usage:
    python train_include_dataset.py --dataset-path /path/to/include/dataset --epochs 50
"""

import argparse
import os
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sign_recognition.models.lstm_model import EnhancedLSTMModel
from sign_recognition.models.mediapipe_extractor import MediaPipeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class INCLUDEDataset(Dataset):
    """
    INCLUDE Dataset Loader for ISL Sign Recognition
    
    Expected directory structure:
    dataset_root/
    ├── sign_class_1/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── sign_class_2/
    │   ├── video1.mp4
    │   └── ...
    └── ...
    """
    
    def __init__(self, dataset_root: str, max_frames: int = 64, 
                 transform=None, cache_features: bool = True):
        self.dataset_root = Path(dataset_root)
        self.max_frames = max_frames
        self.transform = transform
        self.cache_features = cache_features
        
        # Initialize MediaPipe extractor
        self.extractor = MediaPipeExtractor()
        self.feature_dim = self.extractor.total_dim
        
        # Load dataset structure
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset_structure()
        
        # Feature cache
        self.feature_cache = {}
        if cache_features:
            self.cache_dir = self.dataset_root / "feature_cache"
            self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"INCLUDE Dataset loaded:")
        logger.info(f"  Classes: {len(self.class_to_idx)}")
        logger.info(f"  Videos: {len(self.video_paths)}")
        logger.info(f"  Feature dimension: {self.feature_dim}")
    
    def _load_dataset_structure(self):
        """Load the dataset directory structure and create class mappings"""
        class_dirs = [d for d in self.dataset_root.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        # Sort for consistent ordering
        class_dirs.sort(key=lambda x: x.name.lower())
        
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Find all video files in this class directory
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(class_dir.glob(f"*{ext}"))
                video_files.extend(class_dir.glob(f"**/*{ext}"))  # Recursive search
            
            for video_path in video_files:
                self.video_paths.append(video_path)
                self.labels.append(idx)
        
        if not self.video_paths:
            raise ValueError(f"No video files found in {self.dataset_root}")
    
    def _get_cache_path(self, video_path: Path) -> Path:
        """Get cache file path for video features"""
        if not self.cache_features:
            return None
        
        # Create unique cache filename
        relative_path = video_path.relative_to(self.dataset_root)
        cache_name = str(relative_path).replace(os.sep, '_').replace('.', '_') + '.npy'
        return self.cache_dir / cache_name
    
    def _extract_features(self, video_path: Path) -> np.ndarray:
        """Extract features from video file"""
        cache_path = self._get_cache_path(video_path)
        
        # Try to load from cache
        if cache_path and cache_path.exists():
            try:
                features = np.load(cache_path)
                if features.shape == (self.max_frames, self.feature_dim):
                    return features
            except Exception as e:
                logger.warning(f"Failed to load cached features for {video_path}: {e}")
        
        # Extract features using MediaPipe
        try:
            features = self.extractor.extract_video(str(video_path), self.max_frames)
            
            # Save to cache
            if cache_path:
                try:
                    np.save(cache_path, features)
                except Exception as e:
                    logger.warning(f"Failed to cache features for {video_path}: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features from {video_path}: {e}")
            # Return zero features as fallback
            return np.zeros((self.max_frames, self.feature_dim), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract features
        features = self._extract_features(video_path)
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).float()
        
        # Apply transforms if any
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        return features_tensor, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class"""
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'extractor'):
            self.extractor.close()

class INCLUDETrainer:
    """Complete training pipeline for INCLUDE dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        logger.info(f"Trainer initialized - Device: {self.device}")
    
    def load_dataset(self):
        """Load and split the INCLUDE dataset"""
        logger.info("Loading INCLUDE dataset...")
        
        # Load full dataset
        full_dataset = INCLUDEDataset(
            dataset_root=self.config['dataset_path'],
            max_frames=self.config['max_frames'],
            cache_features=self.config['cache_features']
        )
        
        # Print dataset statistics
        distribution = full_dataset.get_class_distribution()
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total videos: {len(full_dataset)}")
        logger.info(f"  Classes: {len(full_dataset.class_to_idx)}")
        logger.info(f"  Feature dimension: {full_dataset.feature_dim}")
        
        # Log class distribution
        logger.info("Class distribution:")
        for class_name, count in sorted(distribution.items()):
            logger.info(f"  {class_name}: {count} videos")
        
        # Split dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Store dataset info
        self.num_classes = len(full_dataset.class_to_idx)
        self.feature_dim = full_dataset.feature_dim
        self.class_to_idx = full_dataset.class_to_idx
        self.idx_to_class = full_dataset.idx_to_class
        
        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(train_dataset)} videos")
        logger.info(f"  Validation: {len(val_dataset)} videos")
        logger.info(f"  Test: {len(test_dataset)} videos")
        
        return full_dataset
    
    def build_model(self):
        """Build the sign recognition model"""
        logger.info("Building model...")
        
        self.model = EnhancedLSTMModel(
            input_dim=self.feature_dim,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_classes=self.num_classes,
            dropout=self.config['dropout'],
            num_heads=self.config['num_heads']
        )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model built:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        logger.info("Setting up training components...")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config['step_size'], gamma=0.1
            )
        elif self.config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        
        logger.info(f"Training setup complete:")
        logger.info(f"  Optimizer: {self.config['optimizer']}")
        logger.info(f"  Learning rate: {self.config['learning_rate']}")
        logger.info(f"  Scheduler: {self.config['scheduler']}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log batch progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f'Epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                           f'({100. * batch_idx / len(self.train_loader):.0f}%)]\\t'
                           f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def test(self) -> Dict:
        """Test the model and generate detailed metrics"""
        self.model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Generate classification report
        class_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy * 100,
            'classification_report': report,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation accuracy: {metrics['val_accuracy']:.2f}%")
    
    def save_metadata(self):
        """Save training metadata"""
        metadata = {
            'dataset_path': str(self.config['dataset_path']),
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'max_frames': self.config['max_frames'],
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'training_config': self.config,
            'training_date': datetime.now().isoformat(),
            'model_architecture': 'EnhancedLSTMModel'
        }
        
        metadata_path = self.output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch}/{self.config['epochs']} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                       f"LR: {current_lr:.6f}, "
                       f"Time: {epoch_time:.1f}s")
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
            
            metrics = {
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }
            
            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best)
            
            # Store for plotting
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_metrics['loss'])
            train_accs.append(train_metrics['accuracy'])
            val_accs.append(val_metrics['accuracy'])
        
        # Final test
        logger.info("Running final test...")
        test_results = self.test()
        logger.info(f"Final test accuracy: {test_results['accuracy']:.2f}%")
        
        # Save final results
        self.save_metadata()
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        self.writer.close()
        logger.info("Training completed!")
        
        return test_results
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {self.output_dir / 'training_curves.png'}")

def create_default_config():
    """Create default training configuration"""
    return {
        # Dataset
        'dataset_path': 'data/include_dataset',
        'max_frames': 64,
        'cache_features': True,
        
        # Model
        'hidden_size': 512,
        'num_layers': 3,
        'dropout': 0.3,
        'num_heads': 8,
        
        # Training
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'step_size': 15,
        'grad_clip': 1.0,
        
        # Logging
        'log_interval': 10,
        'save_interval': 5,
        'output_dir': 'sign_recognition/include_trained_models',
        'num_workers': 4
    }

def main():
    parser = argparse.ArgumentParser(description='Train INCLUDE Dataset for ISL Sign Recognition')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to INCLUDE dataset directory')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='sign_recognition/include_trained_models',
                       help='Output directory for models and logs')
    parser.add_argument('--cache-features', action='store_true', default=True,
                       help='Cache extracted features for faster training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config.update({
        'dataset_path': args.dataset_path,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'output_dir': args.output_dir,
        'cache_features': args.cache_features,
        'num_workers': args.num_workers
    })
    
    # Print configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = INCLUDETrainer(config)
    
    try:
        # Load dataset
        dataset = trainer.load_dataset()
        
        # Build model
        trainer.build_model()
        
        # Setup training
        trainer.setup_training()
        
        # Train
        test_results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {test_results['accuracy']:.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(trainer, 'train_loader') and trainer.train_loader:
            if hasattr(trainer.train_loader.dataset, 'dataset'):
                trainer.train_loader.dataset.dataset.close()

if __name__ == '__main__':
    main()