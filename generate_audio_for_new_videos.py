#!/usr/bin/env python3
"""
Generate TTS audio for new videos that don't have audio files yet
"""

import os
import json
from gtts import gTTS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_missing_audio():
    """Generate TTS audio for videos that don't have audio files"""
    
    # Load metadata
    metadata_path = 'knowledge_base/metadata.json'
    audio_dir = 'knowledge_base/generated_audio'
    
    # Create audio directory if it doesn't exist
    os.makedirs(audio_dir, exist_ok=True)
    
    if not os.path.exists(metadata_path):
        logger.error("metadata.json not found. Run update_metadata.py first!")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Found {len(metadata)} video entries in metadata")
    
    generated_count = 0
    skipped_count = 0
    
    for item in metadata:
        phrase = item['text']
        audio_filename = f"{phrase.replace(' ', '_')}.mp3"
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Check if audio already exists
        if os.path.exists(audio_path):
            skipped_count += 1
            continue
        
        try:
            logger.info(f"Generating audio for: '{phrase}'")
            
            # Generate TTS audio
            tts = gTTS(text=phrase, lang='en', slow=False)
            tts.save(audio_path)
            
            generated_count += 1
            logger.info(f"✅ Generated: {audio_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate audio for '{phrase}': {str(e)}")
    
    logger.info(f"\n🎵 Audio Generation Complete!")
    logger.info(f"Generated: {generated_count} new audio files")
    logger.info(f"Skipped: {skipped_count} existing audio files")
    logger.info(f"Total: {generated_count + skipped_count} audio files available")

if __name__ == "__main__":
    generate_missing_audio()