#!/usr/bin/env python3
"""
Train Sign Recognition Model on Knowledge Base Videos
Uses the videos from knowledge_base/videos/ directory
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_knowledge_base_dataset():
    """
    Prepare dataset from knowledge_base/videos for training
    """
    logger.info("="*70)
    logger.info("PREPARING KNOWLEDGE BASE DATASET FOR TRAINING")
    logger.info("="*70)
    
    # Paths
    videos_dir = Path("knowledge_base/videos")
    metadata_file = Path("knowledge_base/metadata.json")
    output_dir = Path("data/knowledge_base_dataset")
    
    # Check if videos exist
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        return False
    
    # Load metadata
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Found {len(metadata)} videos in metadata")
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Organize videos by sign/phrase
    videos_by_sign = {}
    for entry in metadata:
        text = entry.get('text', '').lower().strip()
        video_file = entry.get('file', '')
        
        if text and video_file:
            if text not in videos_by_sign:
                videos_by_sign[text] = []
            videos_by_sign[text].append(video_file)
    
    logger.info(f"Found {len(videos_by_sign)} unique signs")
    
    # Filter signs with at least 3 videos (needed for train/val/test split)
    valid_signs = {sign: videos for sign, videos in videos_by_sign.items() if len(videos) >= 3}
    
    if len(valid_signs) < 10:
        logger.warning(f"Only {len(valid_signs)} signs have 3+ videos. Need more videos for good training!")
        logger.warning("Consider:")
        logger.warning("1. Using INCLUDE dataset: python complete_include_training.py")
        logger.warning("2. Recording more videos via /contribute page")
        logger.warning("3. Downloading more ISL videos")
    
    logger.info(f"Using {len(valid_signs)} signs with 3+ videos each")
    
    # Split videos into train/val/test
    total_videos = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for sign, videos in valid_signs.items():
        # Create sign directory in each split
        for split in ['train', 'val', 'test']:
            (output_dir / split / sign).mkdir(parents=True, exist_ok=True)
        
        # Split videos: 70% train, 15% val, 15% test
        num_videos = len(videos)
        train_count = max(1, int(num_videos * 0.7))
        val_count = max(1, int(num_videos * 0.15))
        
        train_videos = videos[:train_count]
        val_videos = videos[train_count:train_count + val_count]
        test_videos = videos[train_count + val_count:]
        
        # Copy videos to respective splits
        for split, split_videos in [('train', train_videos), ('val', val_videos), ('test', test_videos)]:
            for video_file in split_videos:
                src = videos_dir / video_file
                if src.exists():
                    dst = output_dir / split / sign / video_file
                    shutil.copy2(src, dst)
                    total_videos += 1
                    split_counts[split] += 1
    
    logger.info(f"\n{'='*70}")
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Total signs: {len(valid_signs)}")
    logger.info(f"Train: {split_counts['train']} videos")
    logger.info(f"Val: {split_counts['val']} videos")
    logger.info(f"Test: {split_counts['test']} videos")
    logger.info(f"Dataset location: {output_dir}")
    
    # Save dataset info
    dataset_info = {
        "name": "Knowledge Base ISL Dataset",
        "created": datetime.now().isoformat(),
        "total_videos": total_videos,
        "total_signs": len(valid_signs),
        "signs": list(valid_signs.keys()),
        "splits": split_counts,
        "source": "knowledge_base/videos"
    }
    
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Create training config
    config = {
        "dataset_config": {
            "dataset_path": str(output_dir),
            "max_frames": 64,
            "cache_features": True
        },
        "model_config": {
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "num_heads": 4
        },
        "training_config": {
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "optimizer": "adamw"
        },
        "logging_config": {
            "output_dir": "sign_recognition/kb_trained_models",
            "tensorboard_dir": "sign_recognition/kb_trained_models/logs"
        }
    }
    
    config_file = Path("kb_training_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("NEXT STEPS")
    logger.info(f"{'='*70}")
    
    if len(valid_signs) >= 10:
        logger.info("✅ Dataset ready for training!")
        logger.info("\nTo train the model, run:")
        logger.info(f"  python train_include_dataset.py --config {config_file}")
        logger.info("\nExpected training time: 30-60 minutes")
        logger.info(f"Expected accuracy: 70-80% (with {len(valid_signs)} signs)")
    else:
        logger.info("⚠️  Dataset too small for good training")
        logger.info("\nRecommendations:")
        logger.info("1. Use INCLUDE dataset (263 signs, 4,292 videos):")
        logger.info("   python complete_include_training.py")
        logger.info("\n2. Or record more videos:")
        logger.info("   - Visit http://127.0.0.1:5000/contribute")
        logger.info("   - Record at least 5 videos per sign")
        logger.info("   - Aim for 20+ different signs")
    
    logger.info(f"{'='*70}\n")
    
    return True

def main():
    print("\n" + "="*70)
    print("KNOWLEDGE BASE DATASET PREPARATION")
    print("Prepare your existing videos for sign recognition training")
    print("="*70 + "\n")
    
    success = prepare_knowledge_base_dataset()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
