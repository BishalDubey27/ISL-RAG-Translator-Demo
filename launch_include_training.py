#!/usr/bin/env python3
"""
INCLUDE Dataset Training Launcher
Complete pipeline for training ISL sign recognition on INCLUDE dataset

This script handles:
1. Dataset preparation and validation
2. Model training with optimal settings
3. Evaluation and testing
4. Model deployment preparation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'include_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print training banner"""
    print("\n" + "="*80)
    print("🎯 INCLUDE DATASET TRAINING PIPELINE")
    print("   Complete ISL Sign Recognition Training System")
    print("="*80)

def check_requirements():
    """Check system requirements for training"""
    logger.info("Checking system requirements...")
    
    requirements = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        requirements.append("❌ Python 3.8+ required")
    else:
        logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'numpy', 'opencv-python', 'mediapipe',
        'scikit-learn', 'matplotlib', 'seaborn', 'tensorboard'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            requirements.append(f"❌ Missing package: {package}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU available: {gpu_name} ({gpu_count} device(s))")
        else:
            logger.info("⚠️ No GPU available - training will use CPU (slower)")
    except ImportError:
        pass
    
    # Check disk space (estimate 50GB needed)
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 50:
            requirements.append(f"⚠️ Low disk space: {free_space:.1f}GB (50GB+ recommended)")
        else:
            logger.info(f"✅ Disk space: {free_space:.1f}GB available")
    except:
        logger.warning("Could not check disk space")
    
    if requirements:
        logger.error("System requirements not met:")
        for req in requirements:
            logger.error(f"  {req}")
        
        if missing_packages:
            logger.info("Install missing packages with:")
            logger.info(f"  pip install {' '.join(missing_packages)}")
        
        return False
    
    logger.info("✅ All system requirements met!")
    return True

def prepare_dataset(source_path: str, target_path: str = "data/include_dataset"):
    """Prepare INCLUDE dataset for training"""
    logger.info("Preparing INCLUDE dataset...")
    
    try:
        from prepare_include_dataset import INCLUDEDatasetPreparer
        
        preparer = INCLUDEDatasetPreparer(source_path, target_path)
        dataset_info = preparer.prepare()
        
        logger.info(f"Dataset prepared successfully:")
        logger.info(f"  Signs: {dataset_info['total_signs']}")
        logger.info(f"  Videos: {dataset_info['total_videos']}")
        logger.info(f"  Location: {target_path}")
        
        return True, target_path
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return False, None

def load_training_config(config_path: str = "include_training_config.json"):
    """Load training configuration"""
    logger.info(f"Loading training configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("Training configuration loaded:")
        logger.info(f"  Epochs: {config['training_config']['epochs']}")
        logger.info(f"  Batch size: {config['training_config']['batch_size']}")
        logger.info(f"  Learning rate: {config['training_config']['learning_rate']}")
        logger.info(f"  Model: {config['model_config']['architecture']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Using default configuration...")
        
        # Return default config
        return {
            "dataset_config": {
                "dataset_path": "data/include_dataset",
                "max_frames": 64,
                "cache_features": True
            },
            "model_config": {
                "hidden_size": 512,
                "num_layers": 3,
                "dropout": 0.3,
                "num_heads": 8
            },
            "training_config": {
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "optimizer": "adamw",
                "scheduler": "cosine"
            },
            "logging_config": {
                "output_dir": "sign_recognition/include_trained_models",
                "save_interval": 5
            },
            "hardware_config": {
                "num_workers": 4
            }
        }

def start_training(config: dict, dataset_path: str):
    """Start the training process"""
    logger.info("Starting INCLUDE dataset training...")
    
    try:
        from train_include_dataset import INCLUDETrainer
        
        # Update dataset path in config
        config['dataset_config']['dataset_path'] = dataset_path
        
        # Flatten config for trainer
        trainer_config = {}
        for section, values in config.items():
            trainer_config.update(values)
        
        # Initialize and run trainer
        trainer = INCLUDETrainer(trainer_config)
        
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
        
        return True, test_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def deploy_model(model_dir: str):
    """Prepare trained model for deployment"""
    logger.info("Preparing model for deployment...")
    
    try:
        model_dir = Path(model_dir)
        
        # Find best model
        best_model_path = model_dir / "checkpoints" / "best_model.pth"
        
        if not best_model_path.exists():
            logger.error(f"Best model not found at {best_model_path}")
            return False
        
        # Copy to deployment location
        deployment_dir = Path("sign_recognition/trained_models")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        
        # Copy best model
        shutil.copy2(best_model_path, deployment_dir / "best_model_include.pth")
        
        # Copy metadata
        metadata_path = model_dir / "training_metadata.json"
        if metadata_path.exists():
            shutil.copy2(metadata_path, deployment_dir / "include_metadata.json")
        
        # Create deployment metadata
        deployment_metadata = {
            "model_name": "INCLUDE ISL Recognition Model",
            "training_date": datetime.now().isoformat(),
            "model_path": "best_model_include.pth",
            "metadata_path": "include_metadata.json",
            "deployment_ready": True
        }
        
        with open(deployment_dir / "deployment_info.json", 'w') as f:
            json.dump(deployment_metadata, f, indent=2)
        
        logger.info("Model deployment preparation completed!")
        logger.info(f"Deployed to: {deployment_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='INCLUDE Dataset Training Pipeline')
    parser.add_argument('--source-dataset', type=str, required=True,
                       help='Path to source INCLUDE dataset')
    parser.add_argument('--target-dataset', type=str, default='data/include_dataset',
                       help='Path for organized dataset')
    parser.add_argument('--config', type=str, default='include_training_config.json',
                       help='Training configuration file')
    parser.add_argument('--skip-preparation', action='store_true',
                       help='Skip dataset preparation (use existing organized dataset)')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare dataset, do not train')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check requirements
    if not check_requirements():
        logger.error("System requirements not met. Please install missing dependencies.")
        return 1
    
    # Load configuration
    config = load_training_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training_config']['epochs'] = args.epochs
    if args.batch_size:
        config['training_config']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training_config']['learning_rate'] = args.learning_rate
    
    # Dataset preparation
    if not args.skip_preparation:
        logger.info("Step 1: Dataset Preparation")
        success, dataset_path = prepare_dataset(args.source_dataset, args.target_dataset)
        if not success:
            logger.error("Dataset preparation failed!")
            return 1
    else:
        dataset_path = args.target_dataset
        if not Path(dataset_path).exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return 1
    
    if args.prepare_only:
        logger.info("Dataset preparation completed. Exiting as requested.")
        return 0
    
    # Training
    logger.info("Step 2: Model Training")
    success, test_results = start_training(config, dataset_path)
    if not success:
        logger.error("Training failed!")
        return 1
    
    # Model deployment
    logger.info("Step 3: Model Deployment")
    output_dir = config['logging_config']['output_dir']
    success = deploy_model(output_dir)
    if not success:
        logger.error("Model deployment failed!")
        return 1
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🎉 INCLUDE DATASET TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Final test accuracy: {test_results['accuracy']:.2f}%")
    logger.info(f"Model deployed to: sign_recognition/trained_models/")
    logger.info("Ready to use in your ISL RAG Translator!")
    logger.info("\nNext steps:")
    logger.info("1. Update unified_app.py to use the new model")
    logger.info("2. Test the model with your sign recognition interface")
    logger.info("3. Deploy to production environment")
    logger.info("="*80)
    
    return 0

if __name__ == '__main__':
    exit(main())