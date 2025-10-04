"""
Simple script to create basic video files for the ISL RAG Demo
This creates minimal video files that can be used for demonstration purposes
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont
import subprocess

def create_simple_videos():
    """Create simple video files based on metadata.json"""
    
    # Read the metadata to get the required video files
    with open('knowledge_base/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create videos directory if it doesn't exist
    output_dir = "knowledge_base/videos/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating simple video files...")
    
    for item in metadata:
        filename = item['file']
        text = item['text']
        
        print(f"Creating: {filename} - '{text}'")
        
        try:
            # Create a simple image with text
            img = Image.new('RGB', (640, 360), color='darkblue')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except:
                font = ImageFont.load_default()
            
            # Get text size and position it in the center
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (640 - text_width) // 2
            y = (360 - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
            # Save as image first
            img_path = f"temp_{filename.replace('.mp4', '.png')}"
            img.save(img_path)
            
            # Convert image to video using ffmpeg (if available)
            video_path = os.path.join(output_dir, filename)
            
            try:
                # Try to use ffmpeg to create a video from the image
                cmd = [
                    'ffmpeg', '-y', '-loop', '1', '-i', img_path, 
                    '-t', '3', '-pix_fmt', 'yuv420p', '-vf', 'scale=640:360',
                    video_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✅ Created: {filename}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If ffmpeg is not available, just copy the image as a placeholder
                print(f"⚠️  FFmpeg not available, creating image placeholder for: {filename}")
                # For now, we'll create a simple text file as placeholder
                with open(video_path.replace('.mp4', '.txt'), 'w') as f:
                    f.write(f"Video placeholder for: {text}")
            
            # Clean up temporary image
            if os.path.exists(img_path):
                os.remove(img_path)
                
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")
    
    print(f"\n🎉 Video files created in {output_dir}")
    print("You can now run the web demo with: python app.py")

if __name__ == "__main__":
    create_simple_videos()
