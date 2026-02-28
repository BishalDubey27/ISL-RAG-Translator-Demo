import os
import json

# Define the paths to your videos folder and metadata file
VIDEOS_DIR = "knowledge_base/videos"
METADATA_FILE = "knowledge_base/metadata.json"

def update_metadata_from_videos():
    """
    Scans the videos directory, finds new .mp4 files, and
    automatically adds them to the metadata.json file.
    """
    print("Starting metadata update...")

    # --- 1. Get the list of video files from the directory ---
    try:
        # List all files in the directory that end with .mp4
        video_files_on_disk = {f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')}
    except FileNotFoundError:
        print(f"Error: The directory '{VIDEOS_DIR}' was not found. Please create it.")
        return

    # --- 2. Load the existing metadata from the JSON file ---
    existing_metadata = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            try:
                existing_metadata = json.load(f)
            except json.JSONDecodeError:
                print("Warning: metadata.json is empty or corrupted. Starting fresh.")
    
    # Create a set of filenames that are already in the metadata for fast lookup
    files_in_metadata = {item['file'] for item in existing_metadata}
    print(f"Found {len(files_in_metadata)} files already cataloged in metadata.json.")

    # --- 3. Find the new files that need to be added ---
    new_files_to_add = video_files_on_disk - files_in_metadata
    
    if not new_files_to_add:
        print("✅ Metadata is already up-to-date. No new videos found.")
        return

    print(f"Found {len(new_files_to_add)} new video(s) to add:")

    # --- 4. Generate new entries and append them ---
    for filename in sorted(list(new_files_to_add)):
        # Automatically generate a clean text phrase from the filename
        # e.g., "how_are_you_sentence.mp4" -> "how are you sentence"
        text_phrase = os.path.splitext(filename)[0].replace('_', ' ')
        
        new_entry = {
            "file": filename,
            "text": text_phrase
        }
        existing_metadata.append(new_entry)
        print(f"  + Added entry for '{filename}' with text '{text_phrase}'")

    # --- 5. Save the updated metadata back to the file ---
    with open(METADATA_FILE, 'w') as f:
        # Use indent=2 for pretty-printing, making the JSON readable
        json.dump(existing_metadata, f, indent=2)

    print(f"\n✅ Successfully updated {METADATA_FILE} with {len(new_files_to_add)} new entries.")

if __name__ == "__main__":
    update_metadata_from_videos()