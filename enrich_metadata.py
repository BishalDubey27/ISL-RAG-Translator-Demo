import os
import json

VIDEO_DIR = "knowledge_base/videos"
METADATA_FILE = "knowledge_base/metadata.json"
SYNONYM_DICT_FILE = "knowledge_base/synonym_dict.json"

# Paraphrase/alternate mappings for each phrase
paraphrase_dict = {
    "how are you": ["how do you do", "what is your well-being", "are you okay",
                    "are you well", "are you fine", "how do you feel"],
    "hello": ["hi", "greetings", "hey", "hello there", "good day"],
    "thank you": ["thanks", "thank you very much", "thanks a lot"],
    "good morning": ["morning", "good day"],
    "good night": ["night", "good evening"],
    "yes": ["yeah", "yep", "correct", "right"],
    "no": ["nope", "not correct", "wrong"],
    # Expand as needed for robust coverage
}

# Synonym dictionary for query preprocessing
synonym_dict = {
    "wellbeing": "well being",
    "what is your well-being": "how are you",
    "okay": "fine",
    "greetings": "hello",
    "hi": "hello",
    "how do you do": "how are you",
    "thanks": "thank you",
    "morning": "good morning",
    "night": "good night",
    "yeah": "yes",
    "yep": "yes",
    "nope": "no",
    # Expand as needed for direct normalization
}

# Helper to convert filename to phrase
def filename_to_phrase(filename):
    """Convert video filename to readable phrase"""
    base = filename.replace(".mp4", "")
    phrase = base.replace("_", " ").strip()
    return phrase

def enrich_metadata():
    """
    Enriches metadata with paraphrases and synonyms for better search performance
    """
    print("Starting metadata enrichment...")
    
    # Load existing metadata if present
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} existing metadata entries")
    else:
        metadata = []
        print("No existing metadata found, creating new")

    # Track all known videos in metadata
    existing_videos = set(entry.get("file", entry.get("video", "")) for entry in metadata)
    new_metadata = list(metadata)
    
    # Check for new videos in the directory
    if os.path.exists(VIDEO_DIR):
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
        print(f"Found {len(video_files)} video files")
        
        for fname in video_files:
            if fname not in existing_videos:
                phrase = filename_to_phrase(fname)
                print(f"Adding new video: {fname} -> '{phrase}'")
                
                # Add main entry
                new_metadata.append({"text": phrase, "file": fname})
                
                # Add paraphrased entries
                para_list = paraphrase_dict.get(phrase, [])
                for para in para_list:
                    new_metadata.append({"text": para, "file": fname})
                    synonym_dict[para] = phrase  # Guarantee preprocessing works correctly
                    print(f"  Added paraphrase: '{para}'")
                
                # Ensure multi-word phrases and their paraphrases are added
                if " " in phrase:
                    print(f"Adding paraphrases for multi-word phrase: '{phrase}'")
                    for para in paraphrase_dict.get(phrase, []):
                        new_metadata.append({"text": para, "file": fname})
    
    # Make metadata unique (avoid duplicates)
    unique_metadata = {}
    for entry in new_metadata:
        key = f"{entry['text'].lower()}|{entry['file']}"
        unique_metadata[key] = entry
    
    final_metadata = list(unique_metadata.values())
    
    # Save enriched metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(final_metadata, f, indent=2)
    
    # Save synonym dictionary for use in the app
    with open(SYNONYM_DICT_FILE, "w") as f:
        json.dump(synonym_dict, f, indent=2)
    
    print(f"✅ Metadata enrichment complete!")
    print(f"   Total entries: {len(final_metadata)}")
    print(f"   Synonym mappings: {len(synonym_dict)}")
    print(f"   Files saved: {METADATA_FILE}, {SYNONYM_DICT_FILE}")

if __name__ == "__main__":
    enrich_metadata()