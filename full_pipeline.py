#!/usr/bin/env python3
"""
Complete ISL Video Processing Pipeline

This script automates the entire workflow:
1. Extract text from ISL screen recordings using OCR
2. Rename videos based on extracted captions
3. Update metadata.json automatically
4. Rebuild the FAISS vector database

Usage:
    python full_pipeline.py
    
    Or with custom directories:
    python full_pipeline.py --input ./my_videos --output ./knowledge_base/videos
"""

import argparse
import subprocess
import sys
from pathlib import Path
from auto_rename_videos import ISLVideoProcessor

class FullISLPipeline:
    def __init__(self, input_dir="staging_videos", output_dir="knowledge_base/videos"):
        self.processor = ISLVideoProcessor(input_dir, output_dir)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
    
    def run_subprocess(self, command, description):
        """Run a subprocess command with proper error handling."""
        print(f"\n🔄 {description}...")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed:")
            print(f"Error: {e.stderr}")
            return False
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            "cv2", "pytesseract", "PIL", "numpy", 
            "sentence_transformers", "faiss", "json"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "cv2":
                    import cv2
                elif package == "PIL":
                    from PIL import Image
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install opencv-python pytesseract pillow numpy sentence-transformers faiss-cpu")
            return False
        
        # Check if tesseract is available
        try:
            subprocess.run(["tesseract", "--version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Tesseract OCR not found. Please install:")
            print("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("Mac: brew install tesseract")
            print("Ubuntu: sudo apt install tesseract-ocr")
            return False
        
        print("✅ All dependencies are available")
        return True
    
    def process_videos(self):
        """Process all videos in the input directory."""
        print(f"\n📁 Processing videos from: {self.input_dir}")
        
        if not self.input_dir.exists():
            print(f"❌ Input directory {self.input_dir} does not exist")
            return False
        
        video_files = list(self.input_dir.glob("*.mp4"))
        if not video_files:
            print(f"ℹ️  No .mp4 files found in {self.input_dir}")
            return True  # Not an error, just nothing to process
        
        print(f"Found {len(video_files)} video(s) to process")
        success = self.processor.process_all_videos()
        return success
    
    def update_metadata(self):
        """Update the metadata.json file."""
        return self.run_subprocess(
            "python update_metadata.py",
            "Updating metadata.json"
        )
    
    def rebuild_database(self):
        """Rebuild the FAISS vector database."""
        return self.run_subprocess(
            "python setup_database.py", 
            "Rebuilding FAISS vector database"
        )
    
    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        print("🚀 Starting Full ISL Video Processing Pipeline")
        print("=" * 50)
        
        steps = [
            ("Dependency Check", self.check_dependencies),
            ("Video Processing", self.process_videos),
            ("Metadata Update", self.update_metadata),
            ("Database Rebuild", self.rebuild_database)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            if not step_func():
                print(f"❌ Pipeline failed at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 Full pipeline completed successfully!")
        print("\n📝 Summary:")
        print(f"   Input folder: {self.input_dir}")
        print(f"   Output folder: {self.output_dir}")
        print(f"   Processed videos moved to: processed_videos/")
        print("\n🚀 Your ISL knowledge base is now updated and ready!")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="ISL Video Processing Pipeline")
    parser.add_argument("--input", "-i", default="staging_videos", 
                       help="Input directory containing ISL screen recordings")
    parser.add_argument("--output", "-o", default="knowledge_base/videos",
                       help="Output directory for processed videos")
    parser.add_argument("--check-deps", action="store_true",
                       help="Only check dependencies and exit")
    
    args = parser.parse_args()
    
    pipeline = FullISLPipeline(args.input, args.output)
    
    if args.check_deps:
        success = pipeline.check_dependencies()
        sys.exit(0 if success else 1)
    
    success = pipeline.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()