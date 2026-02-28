#!/usr/bin/env python3
"""
INCLUDE Dataset Preparation Script
Organizes and prepares the INCLUDE dataset for training

The INCLUDE dataset contains 263 ISL signs with multiple videos per sign.
This script helps organize the dataset into the proper structure for training.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sign_recognition.utils.inference import SignRecognitionInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class INCLUDEDatasetPreparer:
    """Prepares INCLUDE dataset for training"""
    
    def __init__(self, source_path: str, target_path: str):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        
        # INCLUDE dataset sign classes (263 total)
        self.include_signs = [
            # Common words and phrases
            "Hello", "Thank you", "Please", "Sorry", "Yes", "No", "Good", "Bad",
            "Happy", "Sad", "Love", "Help", "Water", "Food", "Home", "School",
            "Work", "Family", "Friend", "Mother", "Father", "Sister", "Brother",
            "Child", "Baby", "Man", "Woman", "Boy", "Girl", "Person", "People",
            
            # Numbers
            "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
            "Eighteen", "Nineteen", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
            "Seventy", "Eighty", "Ninety", "Hundred", "Thousand",
            
            # Colors
            "Red", "Blue", "Green", "Yellow", "Black", "White", "Orange", "Purple",
            "Pink", "Brown", "Gray", "Silver", "Gold",
            
            # Time and dates
            "Today", "Tomorrow", "Yesterday", "Morning", "Afternoon", "Evening", "Night",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December",
            
            # Body parts
            "Head", "Face", "Eye", "Nose", "Mouth", "Ear", "Hair", "Hand", "Finger",
            "Arm", "Leg", "Foot", "Body", "Heart", "Brain",
            
            # Animals
            "Cat", "Dog", "Bird", "Fish", "Cow", "Horse", "Elephant", "Lion", "Tiger",
            "Bear", "Monkey", "Snake", "Rabbit", "Chicken", "Pig",
            
            # Food and drinks
            "Rice", "Bread", "Milk", "Tea", "Coffee", "Sugar", "Salt", "Oil", "Fruit",
            "Apple", "Banana", "Orange", "Mango", "Vegetable", "Onion", "Potato",
            
            # Places
            "House", "Room", "Kitchen", "Bathroom", "Bedroom", "Office", "Hospital",
            "Market", "Shop", "Restaurant", "Hotel", "Airport", "Station", "Park",
            "Temple", "Church", "Mosque", "Library", "University", "College",
            
            # Transportation
            "Car", "Bus", "Train", "Plane", "Bicycle", "Motorcycle", "Boat", "Ship",
            "Taxi", "Auto", "Truck", "Road", "Bridge",
            
            # Actions
            "Go", "Come", "Sit", "Stand", "Walk", "Run", "Jump", "Sleep", "Wake",
            "Eat", "Drink", "Cook", "Clean", "Wash", "Read", "Write", "Study",
            "Play", "Dance", "Sing", "Listen", "Watch", "See", "Look", "Hear",
            "Speak", "Talk", "Tell", "Ask", "Answer", "Think", "Know", "Understand",
            "Remember", "Forget", "Learn", "Teach", "Give", "Take", "Buy", "Sell",
            "Pay", "Cost", "Free", "Open", "Close", "Start", "Stop", "Finish",
            "Begin", "End", "Continue", "Wait", "Hurry", "Slow", "Fast", "Quick",
            
            # Emotions and feelings
            "Angry", "Afraid", "Surprised", "Excited", "Tired", "Sick", "Healthy",
            "Strong", "Weak", "Hot", "Cold", "Warm", "Cool", "Comfortable",
            
            # Communication
            "Name", "Address", "Phone", "Email", "Message", "Letter", "Book",
            "Paper", "Pen", "Pencil", "Computer", "Mobile", "Internet", "Website",
            
            # Weather
            "Sun", "Moon", "Star", "Sky", "Cloud", "Rain", "Snow", "Wind", "Storm",
            "Hot", "Cold", "Warm", "Cool", "Weather",
            
            # Clothing
            "Clothes", "Shirt", "Pant", "Dress", "Shoe", "Hat", "Jacket", "Sari",
            "Kurta", "Jeans", "T-shirt",
            
            # Technology
            "Television", "Radio", "Camera", "Video", "Photo", "Picture", "Movie",
            "Music", "Song", "Sound", "Voice", "Noise", "Quiet", "Loud",
            
            # Medical
            "Doctor", "Nurse", "Medicine", "Hospital", "Sick", "Pain", "Fever",
            "Cough", "Cold", "Headache", "Stomach", "Injection", "Operation",
            
            # Education
            "Student", "Teacher", "Class", "Lesson", "Exam", "Test", "Question",
            "Answer", "Homework", "Subject", "Math", "Science", "History", "English",
            
            # Sports and games
            "Cricket", "Football", "Tennis", "Badminton", "Swimming", "Running",
            "Game", "Play", "Win", "Lose", "Team", "Player", "Ball", "Goal"
        ]
        
        logger.info(f"INCLUDE dataset preparer initialized")
        logger.info(f"Source: {self.source_path}")
        logger.info(f"Target: {self.target_path}")
        logger.info(f"Expected signs: {len(self.include_signs)}")
    
    def scan_source_directory(self) -> Dict[str, List[Path]]:
        """Scan source directory and organize files by sign class"""
        logger.info("Scanning source directory...")
        
        sign_files = {}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.MOV', '.MP4'}
        
        # Recursively find all video files
        for root, dirs, files in os.walk(self.source_path):
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                
                # Check if it's a video file
                if file_path.suffix.lower() in video_extensions:
                    # Try to determine sign class from directory structure or filename
                    sign_class = self._extract_sign_class(file_path)
                    
                    if sign_class:
                        if sign_class not in sign_files:
                            sign_files[sign_class] = []
                        sign_files[sign_class].append(file_path)
        
        logger.info(f"Found {len(sign_files)} sign classes with videos")
        for sign_class, files in sign_files.items():
            logger.info(f"  {sign_class}: {len(files)} videos")
        
        return sign_files
    
    def _extract_sign_class(self, file_path: Path) -> str:
        """Extract sign class from file path or name"""
        # Try to match with known INCLUDE signs
        path_parts = file_path.parts
        
        # Check directory names
        for part in reversed(path_parts[:-1]):  # Exclude filename
            for sign in self.include_signs:
                if sign.lower() in part.lower() or part.lower() in sign.lower():
                    return sign
        
        # Check filename
        filename = file_path.stem.lower()
        for sign in self.include_signs:
            if sign.lower() in filename or filename in sign.lower():
                return sign
        
        # If no match found, use the parent directory name
        if len(path_parts) > 1:
            parent_dir = path_parts[-2]
            # Clean up the directory name
            clean_name = parent_dir.replace('_', ' ').replace('-', ' ').title()
            return clean_name
        
        return None
    
    def organize_dataset(self, sign_files: Dict[str, List[Path]]):
        """Organize files into training structure"""
        logger.info("Organizing dataset...")
        
        # Create target directory
        self.target_path.mkdir(parents=True, exist_ok=True)
        
        total_files = 0
        organized_signs = {}
        
        for sign_class, files in sign_files.items():
            # Create sign class directory
            sign_dir = self.target_path / sign_class
            sign_dir.mkdir(exist_ok=True)
            
            # Copy files
            copied_files = []
            for i, source_file in enumerate(files):
                # Create unique filename
                target_filename = f"{sign_class}_{i+1:03d}{source_file.suffix}"
                target_file = sign_dir / target_filename
                
                try:
                    shutil.copy2(source_file, target_file)
                    copied_files.append(target_file)
                    total_files += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {source_file}: {e}")
            
            if copied_files:
                organized_signs[sign_class] = copied_files
                logger.info(f"Organized {len(copied_files)} videos for '{sign_class}'")
        
        logger.info(f"Dataset organization complete:")
        logger.info(f"  Total signs: {len(organized_signs)}")
        logger.info(f"  Total videos: {total_files}")
        
        return organized_signs
    
    def create_dataset_info(self, organized_signs: Dict[str, List[Path]]):
        """Create dataset information file"""
        dataset_info = {
            "dataset_name": "INCLUDE ISL Dataset",
            "total_signs": len(organized_signs),
            "total_videos": sum(len(files) for files in organized_signs.values()),
            "signs": {}
        }
        
        for sign_class, files in organized_signs.items():
            dataset_info["signs"][sign_class] = {
                "video_count": len(files),
                "video_files": [str(f.relative_to(self.target_path)) for f in files]
            }
        
        # Save dataset info
        info_file = self.target_path / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset info saved to {info_file}")
        
        # Create class list file
        class_list_file = self.target_path / "class_list.txt"
        with open(class_list_file, 'w') as f:
            for sign_class in sorted(organized_signs.keys()):
                f.write(f"{sign_class}\n")
        
        logger.info(f"Class list saved to {class_list_file}")
        
        return dataset_info
    
    def validate_dataset(self, organized_signs: Dict[str, List[Path]]):
        """Validate the organized dataset"""
        logger.info("Validating dataset...")
        
        issues = []
        
        # Check minimum videos per class
        min_videos_per_class = 5
        for sign_class, files in organized_signs.items():
            if len(files) < min_videos_per_class:
                issues.append(f"'{sign_class}' has only {len(files)} videos (minimum: {min_videos_per_class})")
        
        # Check for empty files
        for sign_class, files in organized_signs.items():
            for file_path in files:
                if file_path.stat().st_size == 0:
                    issues.append(f"Empty file: {file_path}")
        
        if issues:
            logger.warning(f"Dataset validation found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more issues")
        else:
            logger.info("Dataset validation passed!")
        
        return issues
    
    def test_model_on_dataset(self, model_path: str, metadata_path: str, model_type: str = 'transformer'):
        """Test the INCLUDE-trained model on the prepared dataset"""
        logger.info("Testing INCLUDE-trained model on the prepared dataset...")

        # Initialize the inference class
        inference = SignRecognitionInference(model_path, metadata_path, model_type=model_type)

        # Iterate through the organized dataset
        dataset_info_path = self.target_path / "dataset_info.json"
        if not dataset_info_path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {dataset_info_path}")

        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

        results = {}
        for sign_class, data in dataset_info['signs'].items():
            logger.info(f"Testing videos for sign class: {sign_class}")
            results[sign_class] = []

            for video_file in data['video_files']:
                video_path = self.target_path / video_file
                try:
                    prediction = inference.predict_video(str(video_path))
                    results[sign_class].append({
                        'video': video_file,
                        'predicted_label': prediction['label'],
                        'confidence': prediction['confidence'],
                        'top5_predictions': prediction['top5_predictions']
                    })
                except Exception as e:
                    logger.error(f"Failed to predict video {video_file}: {e}")

        # Save results
        results_file = self.target_path / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Testing completed. Results saved to {results_file}")

        inference.close()
        return results

    def prepare(self):
        """Main preparation function"""
        logger.info("Starting INCLUDE dataset preparation...")
        
        # Check source directory
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_path}")
        
        # Scan source directory
        sign_files = self.scan_source_directory()
        
        if not sign_files:
            raise ValueError("No video files found in source directory")
        
        # Organize dataset
        organized_signs = self.organize_dataset(sign_files)
        
        # Create dataset info
        dataset_info = self.create_dataset_info(organized_signs)
        
        # Validate dataset
        issues = self.validate_dataset(organized_signs)
        
        logger.info("Dataset preparation completed!")
        logger.info(f"Prepared dataset with {len(organized_signs)} sign classes")
        logger.info(f"Total videos: {sum(len(files) for files in organized_signs.values())}")
        
        if issues:
            logger.warning(f"Found {len(issues)} validation issues - check logs above")
        
        return dataset_info

def main():
    parser = argparse.ArgumentParser(description='Prepare INCLUDE dataset for training')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source INCLUDE dataset directory')
    parser.add_argument('--target', type=str, default='data/include_dataset',
                       help='Path to target organized dataset directory')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing organized dataset')
    parser.add_argument('--test-model', action='store_true',
                       help='Test the INCLUDE-trained model on the dataset')
    parser.add_argument('--model-path', type=str, default='path/to/model.pth',
                       help='Path to the trained model file')
    parser.add_argument('--metadata-path', type=str, default='path/to/metadata.json',
                       help='Path to the metadata file')
    parser.add_argument('--model-type', type=str, default='transformer',
                       help='Type of model to use (e.g., lstm, transformer)')

    args = parser.parse_args()

    preparer = INCLUDEDatasetPreparer(args.source, args.target)

    if args.validate_only:
        # Just validate existing dataset
        if not Path(args.target).exists():
            logger.error(f"Target directory does not exist: {args.target}")
            return
        
        sign_files = preparer.scan_source_directory()
        issues = preparer.validate_dataset(sign_files)
        
        if not issues:
            logger.info("Dataset validation passed!")
        else:
            logger.error(f"Dataset validation failed with {len(issues)} issues")
    elif args.test_model:
        # Test the model on the dataset
        results = preparer.test_model_on_dataset(args.model_path, args.metadata_path, args.model_type)
    else:
        # Full preparation
        try:
            dataset_info = preparer.prepare()
            logger.info("Dataset preparation successful!")
            logger.info(f"Ready to train with: python train_include_dataset.py --dataset-path {args.target}")
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

if __name__ == '__main__':
    main()