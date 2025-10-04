"""
Script to create placeholder video files for the ISL RAG Demo
This creates simple text-based videos that can be used for demonstration purposes
"""

from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
import os

def create_placeholder_videos():
    """Create placeholder video files based on metadata.json"""
    
    # Read the metadata to get the required video files
    import json
    with open('knowledge_base/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create videos directory if it doesn't exist
    output_dir = "knowledge_base/videos/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Video settings
    duration = 3  # seconds
    size = (640, 360)  # width, height
    bg_color = (30, 30, 30)  # Dark background
    text_color = "white"
    font_size = 36
    
    print("Creating placeholder videos...")
    
    for item in metadata:
        filename = item['file']
        text = item['text']
        
        print(f"Creating: {filename} - '{text}'")
        
        try:
            # Create background
            bg = ColorClip(size=size, color=bg_color).set_duration(duration)
            
            # Create text clip
            txt = (TextClip(text, 
                           fontsize=font_size, 
                           color=text_color, 
                           method="caption", 
                           size=(600, None))
                   .set_duration(duration)
                   .set_pos("center"))
            
            # Composite the video
            clip = CompositeVideoClip([bg, txt])
            
            # Write the video file
            output_path = os.path.join(output_dir, filename)
            clip.write_videofile(output_path, 
                               fps=24, 
                               codec="libx264", 
                               audio=False,
                               verbose=False,
                               logger=None)
            
            print(f"✅ Created: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")
    
    print(f"\n🎉 All placeholder videos created in {output_dir}")
    print("You can now run the web demo with: python app.py")

if __name__ == "__main__":
    create_placeholder_videos()

