import os
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

VIDEO_DIR = "knowledge_base/videos"
METADATA_FILE = "knowledge_base/metadata.json"
SYNONYM_DICT_FILE = "knowledge_base/synonym_dict.json"

# Import the enrichment function
from enrich_metadata import enrich_metadata, filename_to_phrase, paraphrase_dict

class VideoFileHandler(FileSystemEventHandler):
    """Handles file system events for video files"""
    
    def __init__(self):
        self.last_update = 0
        self.update_delay = 2  # Wait 2 seconds before processing to avoid multiple triggers
    
    def on_created(self, event):
        if not event.is_dir and event.src_path.endswith('.mp4'):
            print(f"New video detected: {event.src_path}")
            self.schedule_update()
    
    def on_deleted(self, event):
        if not event.is_dir and event.src_path.endswith('.mp4'):
            print(f"Video deleted: {event.src_path}")
            self.schedule_update()
    
    def schedule_update(self):
        """Schedule metadata update with delay to avoid multiple rapid updates"""
        current_time = time.time()
        self.last_update = current_time
        
        # Use threading to delay the update
        import threading
        def delayed_update():
            time.sleep(self.update_delay)
            if time.time() - self.last_update >= self.update_delay:
                self.update_metadata()
        
        threading.Thread(target=delayed_update, daemon=True).start()
    
    def update_metadata(self):
        """Update metadata when videos change"""
        print("Updating metadata due to video changes...")
        try:
            enrich_metadata()
            print("✅ Metadata updated successfully!")
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")

def start_auto_update():
    """Start the automatic update system"""
    if not os.path.exists(VIDEO_DIR):
        print(f"Video directory {VIDEO_DIR} not found. Creating it...")
        os.makedirs(VIDEO_DIR, exist_ok=True)
    
    event_handler = VideoFileHandler()
    observer = Observer()
    observer.schedule(event_handler, VIDEO_DIR, recursive=False)
    
    print(f"🔍 Watching {VIDEO_DIR} for changes...")
    print("Press Ctrl+C to stop")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Auto-update system stopped")
    
    observer.join()

def manual_update():
    """Manually trigger metadata update"""
    print("🔄 Manually updating metadata...")
    enrich_metadata()

def check_system_status():
    """Check the current status of the system"""
    print("📊 System Status:")
    print(f"   Video directory: {VIDEO_DIR}")
    print(f"   Videos found: {len([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]) if os.path.exists(VIDEO_DIR) else 0}")
    
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        print(f"   Metadata entries: {len(metadata)}")
    else:
        print("   Metadata file: Not found")
    
    if os.path.exists(SYNONYM_DICT_FILE):
        with open(SYNONYM_DICT_FILE, 'r') as f:
            synonyms = json.load(f)
        print(f"   Synonym mappings: {len(synonyms)}")
    else:
        print("   Synonym file: Not found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "watch":
            start_auto_update()
        elif command == "update":
            manual_update()
        elif command == "status":
            check_system_status()
        else:
            print("Usage:")
            print("  python auto_update_system.py watch   - Start automatic monitoring")
            print("  python auto_update_system.py update  - Manual update")
            print("  python auto_update_system.py status  - Check system status")
    else:
        print("🚀 ISL RAG Translator - Auto Update System")
        print("\nAvailable commands:")
        print("  watch  - Start automatic monitoring for new videos")
        print("  update - Manually update metadata")
        print("  status - Check current system status")
        print("\nExample: python auto_update_system.py watch")