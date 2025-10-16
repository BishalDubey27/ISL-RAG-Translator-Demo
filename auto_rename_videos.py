import cv2
import os
import re
import shutil
from pathlib import Path
import pytesseract
from PIL import Image
import numpy as np

# Configure tesseract path (adjust for your system if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ISLVideoProcessor:
    def __init__(self, input_dir="staging_videos", output_dir="knowledge_base/videos"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_dir = Path("processed_videos")
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def extract_text_from_video(self, video_path, sample_frames=10):
        """
        Extract text from video frames using OCR.
        Samples multiple frames to get the best text extraction.
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted_texts = []
        
        # Sample frames at regular intervals
        frame_indices = np.linspace(0, frame_count-1, min(sample_frames, frame_count), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Preprocess for better OCR
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use pytesseract to extract text
            try:
                text = pytesseract.image_to_string(thresh, config='--psm 6')
                if text.strip():
                    extracted_texts.append(text.strip())
            except Exception as e:
                print(f"OCR error on frame {frame_idx}: {e}")
                continue
        
        cap.release()
        
        # Process and clean extracted texts
        return self.process_extracted_texts(extracted_texts)
    
    def process_extracted_texts(self, texts):
        """
        Process and clean extracted texts to get the best caption.
        """
        if not texts:
            return None
            
        # Clean and normalize texts
        cleaned_texts = []
        for text in texts:
            # Remove special characters and normalize
            cleaned = re.sub(r'[^\w\s]', '', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if len(cleaned) > 2:  # Only keep meaningful text
                cleaned_texts.append(cleaned)
        
        if not cleaned_texts:
            return None
            
        # Find the most common/consistent text
        # Simple approach: return the most frequent text
        from collections import Counter
        text_counts = Counter(cleaned_texts)
        
        # Return the most common text, or the longest if tie
        if text_counts:
            most_common = text_counts.most_common(1)[0][0]
            return most_common
            
        return None
    
    def generate_safe_filename(self, text):
        """
        Generate a safe filename from extracted text.
        """
        if not text:
            return None
            
        # Remove special characters and normalize
        safe_name = re.sub(r'[^\w\s-]', '', text.lower())
        safe_name = re.sub(r'[-\s]+', '_', safe_name).strip('_')
        
        # Limit length
        safe_name = safe_name[:50] if len(safe_name) > 50 else safe_name
        
        return safe_name if safe_name else None
    
    def process_single_video(self, video_path):
        """
        Process a single video: extract text, rename, and move to output directory.
        """
        print(f"Processing: {video_path.name}")
        
        try:
            # Extract text from video
            extracted_text = self.extract_text_from_video(video_path)
            
            if not extracted_text:
                print(f"⚠️  No text found in {video_path.name}. Skipping...")
                return False
            
            # Generate safe filename
            safe_filename = self.generate_safe_filename(extracted_text)
            
            if not safe_filename:
                print(f"⚠️  Could not generate filename for {video_path.name}. Skipping...")
                return False
            
            # Create new filename with .mp4 extension
            new_filename = f"{safe_filename}.mp4"
            new_path = self.output_dir / new_filename
            
            # Handle filename conflicts
            counter = 1
            while new_path.exists():
                new_filename = f"{safe_filename}_{counter}.mp4"
                new_path = self.output_dir / new_filename
                counter += 1
            
            # Copy video to output directory with new name
            shutil.copy2(video_path, new_path)
            
            # Move original to processed directory
            processed_path = self.processed_dir / video_path.name
            shutil.move(video_path, processed_path)
            
            print(f"✅ Renamed '{video_path.name}' → '{new_filename}'")
            print(f"   Extracted text: '{extracted_text}'")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            return False
    
    def process_all_videos(self):
        """
        Process all videos in the input directory.
        """
        video_files = list(self.input_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"No .mp4 files found in {self.input_dir}")
            return
        
        print(f"Found {len(video_files)} video(s) to process...")
        
        success_count = 0
        for video_file in video_files:
            if self.process_single_video(video_file):
                success_count += 1
        
        print(f"\n✅ Successfully processed {success_count}/{len(video_files)} videos")
        
        if success_count > 0:
            print("\n🔄 Next steps:")
            print("1. Run: python update_metadata.py")
            print("2. Run: python setup_database.py")


def main():
    processor = ISLVideoProcessor()
    processor.process_all_videos()


if __name__ == "__main__":
    main()