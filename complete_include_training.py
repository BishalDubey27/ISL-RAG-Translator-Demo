#!/usr/bin/env python3
"""
Complete INCLUDE Dataset Training Pipeline
Downloads INCLUDE dataset from Zenodo and trains ISL sign recognition model

This script handles the complete pipeline:
1. Downloads 56.8GB INCLUDE dataset from Zenodo
2. Organizes dataset for training
3. Trains MediaPipe + LSTM model on 263 ISL signs
4. Evaluates and deploys the trained model
"""

import os
import sys
import json
import time
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
        logging.FileHandler(f'complete_include_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print training banner"""
    print("\n" + "="*80)
    print("COMPLETE INCLUDE DATASET TRAINING PIPELINE")
    print("   Download + Train + Deploy ISL Recognition System")
    print("   Dataset: 4,292 videos, 263 signs, 56.8 GB")
    print("="*80)

def check_system_requirements():
    """Check system requirements for download and training"""
    logger.info("Checking system requirements...")
    
    requirements = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        requirements.append("Python 3.8+ required")
    else:
        logger.info(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check disk space (need ~100GB for download + extraction + training)
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
        if free_space < 100:
            requirements.append(f"Low disk space: {free_space:.1f}GB (100GB+ recommended)")
        else:
            logger.info(f"Disk space: {free_space:.1f}GB available")
    except:
        logger.warning("Could not check disk space")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'numpy', 'cv2', 'mediapipe',
        'sklearn', 'matplotlib', 'seaborn', 'tensorboard', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"{package}")
        except ImportError:
            missing_packages.append(package)
            requirements.append(f"Missing package: {package}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_name} ({gpu_count} device(s))")
        else:
            logger.info("No GPU available - training will use CPU (much slower)")
    except ImportError:
        pass
    
    if requirements:
        logger.error("System requirements not met:")
        for req in requirements:
            logger.error(f"  {req}")
        
        if missing_packages:
            logger.info("Install missing packages with:")
            logger.info(f"  pip install {' '.join(missing_packages)}")
        
        return False
    
    logger.info("All system requirements met!")
    return True
def download_include_dataset(download_dir: str, organized_dir: str, 
                           categories: list = None) -> bool:
    """Download and organize INCLUDE dataset"""
    logger.info("📥 Step 1: Downloading INCLUDE dataset...")
    
    try:
        from download_include_dataset import INCLUDEDatasetDownloader
        
        downloader = INCLUDEDatasetDownloader(download_dir, organized_dir)
        success = downloader.download_dataset(categories=categories)
        
        if success:
            logger.info("✅ INCLUDE dataset download completed successfully")
            return True
        else:
            logger.error("❌ INCLUDE dataset download failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Dataset download failed: {e}")
        return False

def prepare_dataset_for_training(organized_dir: str, prepared_dir: str) -> bool:
    """Prepare downloaded dataset for training"""
    logger.info("🔧 Step 2: Preparing dataset for training...")
    
    try:
        from prepare_include_dataset import INCLUDEDatasetPreparer
        
        preparer = INCLUDEDatasetPreparer(organized_dir, prepared_dir)
        dataset_info = preparer.prepare()
        
        logger.info("✅ Dataset preparation completed successfully")
        logger.info(f"  Signs: {dataset_info['total_signs']}")
        logger.info(f"  Videos: {dataset_info['total_videos']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {e}")
        return False

def train_include_model(dataset_path: str, config_path: str) -> bool:
    """Train the INCLUDE model"""
    logger.info("🚀 Step 3: Training INCLUDE model...")
    
    try:
        from train_include_dataset import INCLUDETrainer
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update dataset path
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
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"Final test accuracy: {test_results['accuracy']:.2f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def deploy_trained_model(model_dir: str) -> bool:
    """Deploy trained model for use in unified app"""
    logger.info("🚀 Step 4: Deploying trained model...")
    
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
        
        # Update unified app to use new model
        update_unified_app_for_include_model()
        
        logger.info("✅ Model deployment completed successfully!")
        logger.info(f"Deployed to: {deployment_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model deployment failed: {e}")
        return False

def update_unified_app_for_include_model():
    """Update unified_app.py to use the new INCLUDE model"""
    logger.info("🔄 Updating unified app to use INCLUDE model...")
    
    try:
        # Read current unified app
        app_path = Path("unified_app.py")
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Update model path references
        content = content.replace(
            'best_model_mediapipe_lstm.pth',
            'best_model_include.pth'
        )
        content = content.replace(
            'dataset_metadata.json',
            'include_metadata.json'
        )
        
        # Update demo mode message
        content = content.replace(
            'Running sign-to-speech in demo mode',
            'INCLUDE model loaded - 263 ISL signs available'
        )
        
        # Write updated content
        with open(app_path, 'w') as f:
            f.write(content)
        
        logger.info("✅ Unified app updated for INCLUDE model")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not update unified app: {e}")

def create_training_summary(results: dict):
    """Create training summary report"""
    summary = {
        "training_completed": datetime.now().isoformat(),
        "dataset": "INCLUDE ISL Dataset",
        "total_signs": 263,
        "total_videos": 4292,
        "model_architecture": "MediaPipe + Enhanced LSTM",
        "training_results": results,
        "deployment_status": "ready",
        "next_steps": [
            "Test model with unified_app.py",
            "Upload sign videos to test recognition",
            "Use live camera interface for real-time recognition"
        ]
    }
    
    summary_file = Path("include_training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📊 Training summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Complete INCLUDE Dataset Training Pipeline')
    parser.add_argument('--download-dir', type=str, default='data/include_raw',
                       help='Directory to download zip files (default: data/include_raw)')
    parser.add_argument('--organized-dir', type=str, default='data/include_organized',
                       help='Directory for organized dataset (default: data/include_organized)')
    parser.add_argument('--prepared-dir', type=str, default='data/include_dataset',
                       help='Directory for training-ready dataset (default: data/include_dataset)')
    parser.add_argument('--config', type=str, default='include_training_config.json',
                       help='Training configuration file')
    parser.add_argument('--categories', nargs='+',
                       help='Specific categories to download (e.g., Animals Greetings)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download (use existing data)')
    parser.add_argument('--skip-preparation', action='store_true',
                       help='Skip dataset preparation (use existing prepared data)')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download dataset, do not train')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model (skip download and preparation)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements not met. Please install missing dependencies.")
        return 1
    
    # Estimate time and space requirements
    logger.info("📊 Pipeline Requirements:")
    logger.info("  Download time: ~2-4 hours (depending on internet speed)")
    logger.info("  Training time: ~4-8 hours (depending on GPU)")
    logger.info("  Disk space needed: ~100 GB")
    logger.info("  Final model size: ~50 MB")
    
    start_time = time.time()
    
    try:
        # Step 1: Download dataset
        if not args.skip_download and not args.train_only:
            if not download_include_dataset(args.download_dir, args.organized_dir, args.categories):
                return 1
        
        if args.download_only:
            logger.info("✅ Download completed. Exiting as requested.")
            return 0
        
        # Step 2: Prepare dataset for training
        if not args.skip_preparation and not args.train_only:
            if not prepare_dataset_for_training(args.organized_dir, args.prepared_dir):
                return 1
        
        # Step 3: Train model
        if not train_include_model(args.prepared_dir, args.config):
            return 1
        
        # Step 4: Deploy model
        output_dir = "sign_recognition/include_trained_models"
        if not deploy_trained_model(output_dir):
            return 1
        
        # Create summary
        training_results = {
            "status": "completed",
            "duration_hours": (time.time() - start_time) / 3600
        }
        create_training_summary(training_results)
        
        # Final success message
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("🎉 COMPLETE INCLUDE TRAINING PIPELINE SUCCESSFUL!")
        logger.info("="*80)
        logger.info(f"⏱️ Total time: {total_time/3600:.1f} hours")
        logger.info(f"🎯 Model trained on 263 ISL signs from 4,292 videos")
        logger.info(f"📁 Model deployed to: sign_recognition/trained_models/")
        logger.info("🚀 Ready to use with unified_app.py!")
        logger.info("\n🎬 Next steps:")
        logger.info("1. python unified_app.py  # Start the complete ISL system")
        logger.info("2. Visit http://127.0.0.1:5000/sign-recognition")
        logger.info("3. Upload ISL videos to test recognition")
        logger.info("4. Use live camera interface for real-time recognition")
        logger.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())