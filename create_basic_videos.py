"""
Create basic video files using OpenCV or simple image sequences
"""

import os
import json
import cv2
import numpy as np

def create_basic_videos():
    """Create basic video files using OpenCV"""
    
    # Read the metadata
    with open('knowledge_base/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create videos directory
    output_dir = "knowledge_base/videos/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating basic video files...")
    
    for item in metadata:
        filename = item['file']
        text = item['text']
        
        print(f"Creating: {filename} - '{text}'")
        
        try:
            # Video properties
            width, height = 640, 360
            fps = 30
            duration = 3  # seconds
            total_frames = fps * duration
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(output_dir, filename)
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Create frames
            for frame_num in range(total_frames):
                # Create a dark blue background
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (30, 30, 100)  # Dark blue background
                
                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                color = (255, 255, 255)  # White text
                thickness = 2
                
                # Get text size for centering
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                
                # Add text to frame
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
                
                # Write frame
                out.write(frame)
            
            # Release video writer
            out.release()
            print(f"✅ Created: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")
    
    print(f"\n🎉 Video files created in {output_dir}")

if __name__ == "__main__":
    create_basic_videos()
