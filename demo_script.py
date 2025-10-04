import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- 1. Load all necessary components ---
print("Loading components for the demo...")

# Load the FAISS index from disk
index = faiss.read_index('video_index.faiss')

# Load the mapping from index ID to filename
with open('index_map.json', 'r') as f:
    index_to_filename = json.load(f)
    # JSON keys are strings, so convert them back to integers for proper mapping
    index_to_filename = {int(k): v for k, v in index_to_filename.items()}

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Components loaded successfully.")


# --- 2. Define the Query and Search Function ---
def get_video_playlist(query_text, k=1):
    """
    Takes a text query, embeds it, and searches the FAISS index.
    Returns the filenames and corresponding text of the top k most similar videos.
    """
    print(f"\nSearching for query: '{query_text}'")
    
    # Load metadata to get text for each video
    with open('knowledge_base/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create a mapping from filename to text
    filename_to_text = {item['file']: item['text'] for item in metadata}
    
    # Encode the query text into a vector using the same model
    query_vector = model.encode([query_text], convert_to_numpy=True)
    
    # Perform the search in the FAISS index
    # D contains the distances (similarity scores), I contains the indices of the top k results
    distances, indices = index.search(query_vector, k)
    
    # --- 3. Retrieve and Display Results ---
    results = []
    print("\n--- Search Results ---")
    if not indices.size or indices[0][0] == -1:
        print("No results found.")
        return []

    for i in range(len(indices[0])):
        idx = indices[0][i]
        filename = index_to_filename.get(idx, "Unknown")
        text = filename_to_text.get(filename, "Text not available")
        distance = distances[0][i]
        similarity_score = 1 / (1 + distance)  # Convert distance to similarity
        results.append({'filename': filename, 'text': text})
        
        print(f"Rank {i+1}:")
        print(f"  - Video File: {filename}")
        print(f"  - Spoken Text: \"{text}\"")
        print(f"  - Similarity Score: {similarity_score:.4f}")
        print(f"  - Distance: {distance:.4f}")

    return results


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Example 1: A query that should have a direct match in metadata.json
    query1 = "how are you"
    playlist1 = get_video_playlist(query1, k=1)
    print(f"\n==> Recommended Playlist for '{query1}': {playlist1}")

    print("\n" + "="*50 + "\n")

    # Example 2: A query that is semantically similar but not an exact match
    # The model should find "How are you?" as the closest phrase.
    query2 = "what is your well-being"
    playlist2 = get_video_playlist(query2, k=1)
    print(f"\n==> Recommended Playlist for '{query2}': {playlist2}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: A query where you might want more than one result
    # "greetings" is semantically close to "Hello" and "Good morning"
    query3 = "greetings"
    playlist3 = get_video_playlist(query3, k=2)
    print(f"\n==> Recommended Playlist for '{query3}': {playlist3}")