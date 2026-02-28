#!/usr/bin/env python3
"""
Minimal INCLUDE Training - Small Real Dataset
Downloads just 5-10 videos per category for quick training

This script:
- Downloads only 1 category zip file (~1-2 GB instead of 56.8 GB)
- Extracts just 5-10 videos per sign
- Trains on ~100-200 videos total
- Completes in 1-2 hours total
- Uses real INCLUDE dataset videos
"""

import os
import sys
import json
import time
import zipfile
import logging
import requests
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinimalINCLUDETrainer:
    """Minimal training with small subset of real INCLUDE data"""
    
    def __init__(self):
        self.download_dir = Path("data/include_minimal")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Select just 2-3 small category files
        self.selected_files = [
            {
                "name": "Greetings_1of2.zip",
                "size": "1.6 GB",
                "url": "https://zenodo.org/records/4010759/files/Greetings_1of2.zip",
                "category": "Greetings"
            },
            # Add more if needed, but start with just one
        ]
        
        self.videos_per_sign = 5  # Only use 5 videos per sign
        
        logger.info("Minimal INCLUDE Trainer initialized")
        logger.info(f"Will download {len(self.selected_files)} category file(s)")
        logger.info(f"Using {self.videos_per_sign} videos per sign")
    
    def download_category(self, file_info):
        """Download a single category file"""
        filename = file_info['name']
        url = file_info['url']
        file_path = self.download_dir / filename
        
        if file_path.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return True
        
        logger.info(f"Downloading {filename} ({file_info['size']})...")
        logger.info("This will take 10-30 minutes depending on your internet speed")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress every 50MB
                        if downloaded % (50 * 1024 * 1024) < 8192:
                            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                            logger.info(f"  Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
            
            logger.info(f"Download complete: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def extract_limited_videos(self, zip_path, max_videos_per_sign=5):
        """Extract only a limited number of videos per sign"""
        logger.info(f"Extracting limited videos from {zip_path.name}...")
        
        extract_dir = self.download_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        extracted_count = 0
        sign_video_counts = {}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all video files
                video_files = [f for f in zip_ref.namelist() 
                              if f.lower().endswith(('.mp4', '.mov', '.avi'))]
                
                logger.info(f"Found {len(video_files)} videos in archive")
                logger.info(f"Extracting max {max_videos_per_sign} videos per sign...")
                
                for file_path in video_files:
                    # Extract sign name from path
                    parts = Path(file_path).parts
                    if len(parts) >= 2:
                        sign_name = parts[-2]  # Parent directory is sign name
                        
                        # Track count per sign
                        if sign_name not in sign_video_counts:
                            sign_video_counts[sign_name] = 0
                        
                        # Only extract if under limit
                        if sign_video_counts[sign_name] < max_videos_per_sign:
                            zip_ref.extract(file_path, extract_dir)
                            sign_video_counts[sign_name] += 1
                            extracted_count += 1
                            
                            if extracted_count % 10 == 0:
                                logger.info(f"  Extracted {extracted_count} videos...")
            
            logger.info(f"Extraction complete: {extracted_count} videos from {len(sign_video_counts)} signs")
            
            return extract_dir, extracted_count, sign_video_counts
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return None, 0, {}
    
    def organize_for_training(self, extract_dir, sign_counts):
        """Organize extracted videos for training"""
        logger.info("Organizing videos for training...")
        
        organized_dir = Path("data/include_minimal_organized")
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            (organized_dir / split).mkdir(exist_ok=True)
        
        total_organized = 0
        
        # Walk through extracted files
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.avi')):
                    source_path = Path(root) / file
                    
                    # Get sign name from parent directory
                    sign_name = Path(root).name
                    
                    # Determine split (70% train, 15% val, 15% test)
                    video_num = total_organized % 10
                    if video_num < 7:
                        split = 'train'
                    elif video_num < 8:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    # Create sign directory in split
                    sign_dir = organized_dir / split / sign_name
                    sign_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy video
                    dest_path = sign_dir / file
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(source_path, dest_path)
                        total_organized += 1
        
        logger.info(f"Organization complete: {total_organized} videos organized")
        
        # Save dataset info
        dataset_info = {
            "name": "INCLUDE Minimal Dataset",
            "total_videos": total_organized,
            "total_signs": len(sign_counts),
            "videos_per_sign": self.videos_per_sign,
            "splits": {
                "train": len(list((organized_dir / "train").rglob("*.mp4"))),
                "val": len(list((organized_dir / "val").rglob("*.mp4"))),
                "test": len(list((organized_dir / "test").rglob("*.mp4")))
            },
            "created": datetime.now().isoformat()
        }
        
        with open(organized_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset info saved: {organized_dir / 'dataset_info.json'}")
        
        return organized_dir, dataset_info
    
    def train_model(self, dataset_path):
        """Train model on minimal dataset"""
        logger.info("Training model on minimal dataset...")
        logger.info("This will use the existing train_include_dataset.py script")
        
        # Update config for minimal training
        config = {
            "dataset_config": {
                "dataset_path": str(dataset_path),
                "max_frames": 64,
                "cache_features": True,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15
            },
            "model_config": {
                "hidden_size": 256,  # Smaller model
                "num_layers": 2,
                "dropout": 0.3,
                "num_heads": 4
            },
            "training_config": {
                "epochs": 20,  # Fewer epochs
                "batch_size": 8,
                "learning_rate": 1e-4,
                "optimizer": "adamw"
            }
        }
        
        # Save minimal config
        config_path = Path("minimal_training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved: {config_path}")
        logger.info("\nTo train the model, run:")
        logger.info(f"  python train_include_dataset.py --config {config_path}")
        logger.info("\nOr continue with automated training...")
        
        # Try to import and run training
        try:
            logger.info("Attempting automated training...")
            
            # Import training module
            sys.path.insert(0, str(Path.cwd()))
            from train_include_dataset import INCLUDETrainer
            
            # Flatten config
            trainer_config = {}
            for section, values in config.items():
                trainer_config.update(values)
            
            # Initialize trainer
            trainer = INCLUDETrainer(trainer_config)
            
            # Load dataset
            logger.info("Loading dataset...")
            dataset = trainer.load_dataset()
            
            # Build model
            logger.info("Building model...")
            trainer.build_model()
            
            # Setup training
            logger.info("Setting up training...")
            trainer.setup_training()
            
            # Train
            logger.info("Starting training...")
            test_results = trainer.train()
            
            logger.info(f"Training complete! Test accuracy: {test_results['accuracy']:.2f}%")
            
            return True
            
        except ImportError as e:
            logger.warning(f"Could not import training module: {e}")
            logger.info("Please run training manually with the command above")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self):
        """Run the complete minimal training pipeline"""
        logger.info("\n" + "="*70)
        logger.info("MINIMAL INCLUDE TRAINING PIPELINE")
        logger.info("Small subset of real INCLUDE data for quick training")
        logger.info("="*70)
        
        start_time = time.time()
        
        try:
            # Step 1: Download category files
            logger.info("\nStep 1: Downloading category files...")
            for file_info in self.selected_files:
                if not self.download_category(file_info):
                    logger.error("Download failed")
                    return False
            
            # Step 2: Extract limited videos
            logger.info("\nStep 2: Extracting limited videos...")
            for file_info in self.selected_files:
                zip_path = self.download_dir / file_info['name']
                extract_dir, count, sign_counts = self.extract_limited_videos(
                    zip_path, self.videos_per_sign
                )
                
                if extract_dir is None:
                    logger.error("Extraction failed")
                    return False
            
            # Step 3: Organize for training
            logger.info("\nStep 3: Organizing for training...")
            organized_dir, dataset_info = self.organize_for_training(extract_dir, sign_counts)
            
            # Step 4: Train model
            logger.info("\nStep 4: Training model...")
            logger.info(f"Dataset: {dataset_info['total_videos']} videos, {dataset_info['total_signs']} signs")
            
            # Show training command
            elapsed = time.time() - start_time
            logger.info("\n" + "="*70)
            logger.info("DATA PREPARATION COMPLETE!")
            logger.info("="*70)
            logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
            logger.info(f"Dataset location: {organized_dir}")
            logger.info(f"Total videos: {dataset_info['total_videos']}")
            logger.info(f"Total signs: {dataset_info['total_signs']}")
            logger.info("\nTo train the model, run:")
            logger.info("  python train_include_dataset.py --config minimal_training_config.json")
            logger.info("\nExpected training time: 30-60 minutes")
            logger.info("="*70)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Minimal INCLUDE Training')
    parser.add_argument('--videos-per-sign', type=int, default=5,
                       help='Number of videos to use per sign (default: 5)')
    parser.add_argument('--auto-train', action='store_true',
                       help='Automatically start training after data preparation')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MINIMAL INCLUDE TRAINING")
    print("Quick training with small subset of real INCLUDE data")
    print("="*70)
    print(f"\nDownload size: ~1.6 GB (instead of 56.8 GB)")
    print(f"Videos per sign: {args.videos_per_sign}")
    print(f"Total time: ~1-2 hours (download + training)")
    print("="*70 + "\n")
    
    trainer = MinimalINCLUDETrainer()
    trainer.videos_per_sign = args.videos_per_sign
    
    success = trainer.run_pipeline()
    
    if success and args.auto_train:
        logger.info("\nStarting automated training...")
        trainer.train_model(Path("data/include_minimal_organized"))
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
