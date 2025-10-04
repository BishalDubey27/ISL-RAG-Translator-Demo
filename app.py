from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Global variables for the model and index
model = None
index = None
index_to_filename = None

def load_components():
    """Load the FAISS index and model components"""
    global model, index, index_to_filename
    
    try:
        # Load the FAISS index from disk
        index = faiss.read_index('video_index.faiss')
        
        # Load the mapping from index ID to filename
        with open('index_map.json', 'r') as f:
            index_to_filename = json.load(f)
            # JSON keys are strings, so convert them back to integers for proper mapping
            index_to_filename = {int(k): v for k, v in index_to_filename.items()}
        
        # Load the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return True
    except Exception as e:
        print(f"Error loading components: {e}")
        return False

def get_video_playlist(query_text, k=3):
    """
    Takes a text query, embeds it, and searches the FAISS index.
    For compound phrases, splits them and finds multiple relevant videos.
    Returns the filenames and corresponding text of the top k most similar videos.
    """
    if model is None or index is None:
        return []
    
    try:
        # Load metadata to get text for each video
        with open('knowledge_base/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create a mapping from filename to text
        filename_to_text = {item['file']: item['text'] for item in metadata}
        
        # Check if this is a compound phrase (contains comma, multiple words, etc.)
        is_compound = ',' in query_text or len(query_text.split()) > 3
        
        if is_compound:
            # For compound phrases, search for individual components
            return search_compound_phrase(query_text, filename_to_text, k)
        else:
            # For simple phrases, use the original single search
            return search_single_phrase(query_text, filename_to_text, k)
            
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def search_single_phrase(query_text, filename_to_text, k):
    """Search for a single phrase"""
    # Encode the query text into a vector using the same model
    query_vector = model.encode([query_text], convert_to_numpy=True)
    
    # Perform the search in the FAISS index
    distances, indices = index.search(query_vector, k)
    
    results = []
    if indices.size and indices[0][0] != -1:
        for i in range(len(indices[0])):
            idx = indices[0][i]
            filename = index_to_filename.get(idx, "Unknown")
            text = filename_to_text.get(filename, "Text not available")
            distance = distances[0][i]
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity
            results.append({
                'filename': filename,
                'text': text,
                'similarity': float(round(similarity_score, 3)),
                'distance': float(round(distance, 3))
            })
    
    return results

def search_compound_phrase(query_text, filename_to_text, k):
    """Search for compound phrases by splitting and finding multiple relevant videos"""
    # Load metadata again to get the list
    with open('knowledge_base/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Get all available phrases from metadata
    available_phrases = [item['text'].lower() for item in metadata]
    
    # Find the best matching phrases
    best_matches = []
    used_filenames = set()
    
    # First, try to find exact phrase matches
    query_lower = query_text.lower()
    for item in metadata:
        phrase_lower = item['text'].lower()
        # Check if the phrase is contained in the query
        if phrase_lower in query_lower and item['file'] not in used_filenames:
            best_matches.append({
                'filename': item['file'],
                'text': item['text'],
                'similarity': 1.0,  # Exact match
                'distance': 0.0,
                'match_type': 'exact'
            })
            used_filenames.add(item['file'])
    
    # If we don't have enough matches, use semantic search for the entire query
    if len(best_matches) < k:
        remaining_k = k - len(best_matches)
        
        # Encode and search the entire query
        query_vector = model.encode([query_text], convert_to_numpy=True)
        distances, indices = index.search(query_vector, remaining_k)
        
        if indices.size and indices[0][0] != -1:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                filename = index_to_filename.get(idx, "Unknown")
                
                if filename not in used_filenames:
                    text = filename_to_text.get(filename, "Text not available")
                    distance = distances[0][i]
                    similarity_score = 1 / (1 + distance)
                    best_matches.append({
                        'filename': filename,
                        'text': text,
                        'similarity': float(round(similarity_score, 3)),
                        'distance': float(round(distance, 3)),
                        'match_type': 'semantic'
                    })
                    used_filenames.add(filename)
    
    # Sort by similarity (exact matches first, then by similarity score)
    best_matches.sort(key=lambda x: (x['match_type'] != 'exact', -x['similarity']))
    
    return best_matches[:k]

@app.route('/')
def index_page():
    """Main page with the demo interface"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """API endpoint for searching videos"""
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a query'}), 400
    
    # Determine if this is a compound phrase
    is_compound = ',' in query or len(query.split()) > 3
    k = 3 if is_compound else 1  # Return multiple videos for compound phrases
    
    results = get_video_playlist(query, k=k)
    return jsonify({'results': results, 'is_compound': is_compound})

@app.route('/videos/<filename>')
def serve_video(filename):
    """Serve video files from the knowledge_base/videos directory"""
    return send_from_directory('knowledge_base/videos', filename)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("Loading ISL RAG Translator Demo...")
    
    # Check if the database exists
    if not os.path.exists('video_index.faiss') or not os.path.exists('index_map.json'):
        print("❌ Database not found. Please run 'python setup_database.py' first.")
        exit(1)
    
    # Load components
    if load_components():
        print("✅ Components loaded successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load components. Please check your setup.")
        exit(1)

