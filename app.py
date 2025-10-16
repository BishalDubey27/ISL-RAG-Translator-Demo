import os
import json
import string
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, jsonify, request, send_from_directory

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- Global variables to hold the loaded components ---
model = None
metadata_list = None
known_phrases_sorted = None
text_to_file_map = None

# --- 2. Load AI Components ---
def load_components():
    """
    Loads the AI model and all necessary mappings into memory.
    """
    global model, metadata_list, known_phrases_sorted, text_to_file_map
    
    print("Loading AI components...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
        
    meta_path = 'knowledge_base/metadata.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}.")
    with open(meta_path, 'r') as f:
        metadata_list = json.load(f)

    # Create helper mappings for the new logic
    # Sort phrases by length (longest first) to ensure greedy matching works correctly
    known_phrases_sorted = sorted([item['text'].lower() for item in metadata_list], key=len, reverse=True)
    # Direct mapping from a known text phrase to its filename
    text_to_file_map = {item['text'].lower(): item['file'] for item in metadata_list}

    print("✅ AI components loaded successfully.")

# --- 3. The Final "Longest Match First" Search Logic ---
def get_video_playlist(query_text):
    """
    Final Logic: Iteratively finds the longest possible exact phrase from the 
    start of the query until the query is fully processed.
    """
    print(f"\nPerforming 'longest match first' search for query: '{query_text}'")
    
    playlist = []
    # Clean the query by replacing punctuation with spaces and making it lowercase
    space_maker = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    remaining_query = query_text.lower().translate(space_maker).strip()

    while len(remaining_query) > 0:
        found_match = False
        
        # Iterate through our known phrases, from longest to shortest
        for phrase in known_phrases_sorted:
            # Check if the remaining query starts with this phrase
            if remaining_query.startswith(phrase):
                # Check for a word boundary to prevent partial matches (e.g., matching "art" in "start")
                if len(remaining_query) == len(phrase) or remaining_query[len(phrase)] == ' ':
                    print(f"  - Match found: '{phrase}'")
                    filename = text_to_file_map[phrase]
                    playlist.append({"file": filename, "text": phrase, "score": 0.0})
                    
                    # Update the remaining query by removing the matched phrase
                    remaining_query = remaining_query[len(phrase):].strip()
                    found_match = True
                    break # Restart the search with the rest of the query
        
        # If no known phrase (long or short) was found at the beginning,
        # we must remove the first word to avoid an infinite loop.
        if not found_match:
            words = remaining_query.split()
            if not words: break
            
            first_word = words[0]
            print(f"  - No match found for '{first_word}'. Skipping.")
            remaining_query = remaining_query[len(first_word):].strip()

    return playlist

# --- 4. Define Web Routes ---
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "Empty query"}), 400
    playlist = get_video_playlist(query)
    return jsonify({"playlist": playlist})

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('knowledge_base/videos', filename)

# --- 5. Run the Application ---
if __name__ == '__main__':
    load_components()
    app.run(host='0.0.0.0', port=5000, debug=True)

