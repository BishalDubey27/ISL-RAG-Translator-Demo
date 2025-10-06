# # # from flask import Flask, render_template, request, jsonify, send_from_directory
# # # import json
# # # import numpy as np
# # # import faiss
# # # from sentence_transformers import SentenceTransformer
# # # import os

# # # app = Flask(__name__)

# # # # Global variables for the model and index
# # # model = None
# # # index = None
# # # index_to_filename = None

# # # def load_components():
# # #     """Load the FAISS index and model components"""
# # #     global model, index, index_to_filename
    
# # #     try:
# # #         # Load the FAISS index from disk
# # #         index = faiss.read_index('video_index.faiss')
        
# # #         # Load the mapping from index ID to filename
# # #         with open('index_map.json', 'r') as f:
# # #             index_to_filename = json.load(f)
# # #             # JSON keys are strings, so convert them back to integers for proper mapping
# # #             index_to_filename = {int(k): v for k, v in index_to_filename.items()}
        
# # #         # Load the sentence transformer model
# # #         model = SentenceTransformer('all-MiniLM-L6-v2')
        
# # #         return True
# # #     except Exception as e:
# # #         print(f"Error loading components: {e}")
# # #         return False

# # # def get_video_playlist(query_text, k=3):
# # #     """
# # #     Takes a text query, embeds it, and searches the FAISS index.
# # #     For compound phrases, splits them and finds multiple relevant videos.
# # #     Returns the filenames and corresponding text of the top k most similar videos.
# # #     """
# # #     if model is None or index is None:
# # #         return []
    
# # #     try:
# # #         # Load metadata to get text for each video
# # #         with open('knowledge_base/metadata.json', 'r') as f:
# # #             metadata = json.load(f)
        
# # #         # Create a mapping from filename to text
# # #         filename_to_text = {item['file']: item['text'] for item in metadata}
        
# # #         # Check if this is a compound phrase (contains comma, multiple words, etc.)
# # #         is_compound = ',' in query_text or len(query_text.split()) > 3
        
# # #         if is_compound:
# # #             # For compound phrases, search for individual components
# # #             return search_compound_phrase(query_text, filename_to_text, k)
# # #         else:
# # #             # For simple phrases, use the original single search
# # #             return search_single_phrase(query_text, filename_to_text, k)
            
# # #     except Exception as e:
# # #         print(f"Error in search: {e}")
# # #         return []

# # # def search_single_phrase(query_text, filename_to_text, k):
# # #     """Search for a single phrase"""
# # #     # Encode the query text into a vector using the same model
# # #     query_vector = model.encode([query_text], convert_to_numpy=True)
    
# # #     # Perform the search in the FAISS index
# # #     distances, indices = index.search(query_vector, k)
    
# # #     results = []
# # #     if indices.size and indices[0][0] != -1:
# # #         for i in range(len(indices[0])):
# # #             idx = indices[0][i]
# # #             filename = index_to_filename.get(idx, "Unknown")
# # #             text = filename_to_text.get(filename, "Text not available")
# # #             distance = distances[0][i]
# # #             similarity_score = 1 / (1 + distance)  # Convert distance to similarity
# # #             results.append({
# # #                 'filename': filename,
# # #                 'text': text,
# # #                 'similarity': float(round(similarity_score, 3)),
# # #                 'distance': float(round(distance, 3))
# # #             })
    
# # #     return results

# # # def search_compound_phrase(query_text, filename_to_text, k):
# # #     """Search for compound phrases by splitting and finding multiple relevant videos"""
# # #     # Load metadata again to get the list
# # #     with open('knowledge_base/metadata.json', 'r') as f:
# # #         metadata = json.load(f)
    
# # #     # Get all available phrases from metadata
# # #     available_phrases = [item['text'].lower() for item in metadata]
    
# # #     # Find the best matching phrases
# # #     best_matches = []
# # #     used_filenames = set()
    
# # #     # First, try to find exact phrase matches
# # #     query_lower = query_text.lower()
# # #     for item in metadata:
# # #         phrase_lower = item['text'].lower()
# # #         # Check if the phrase is contained in the query
# # #         if phrase_lower in query_lower and item['file'] not in used_filenames:
# # #             best_matches.append({
# # #                 'filename': item['file'],
# # #                 'text': item['text'],
# # #                 'similarity': 1.0,  # Exact match
# # #                 'distance': 0.0,
# # #                 'match_type': 'exact'
# # #             })
# # #             used_filenames.add(item['file'])
    
# # #     # If we don't have enough matches, use semantic search for the entire query
# # #     if len(best_matches) < k:
# # #         remaining_k = k - len(best_matches)
        
# # #         # Encode and search the entire query
# # #         query_vector = model.encode([query_text], convert_to_numpy=True)
# # #         distances, indices = index.search(query_vector, remaining_k)
        
# # #         if indices.size and indices[0][0] != -1:
# # #             for i in range(len(indices[0])):
# # #                 idx = indices[0][i]
# # #                 filename = index_to_filename.get(idx, "Unknown")
                
# # #                 if filename not in used_filenames:
# # #                     text = filename_to_text.get(filename, "Text not available")
# # #                     distance = distances[0][i]
# # #                     similarity_score = 1 / (1 + distance)
# # #                     best_matches.append({
# # #                         'filename': filename,
# # #                         'text': text,
# # #                         'similarity': float(round(similarity_score, 3)),
# # #                         'distance': float(round(distance, 3)),
# # #                         'match_type': 'semantic'
# # #                     })
# # #                     used_filenames.add(filename)
    
# # #     # Sort by similarity (exact matches first, then by similarity score)
# # #     best_matches.sort(key=lambda x: (x['match_type'] != 'exact', -x['similarity']))
    
# # #     return best_matches[:k]

# # # @app.route('/')
# # # def index_page():
# # #     """Main page with the demo interface"""
# # #     return render_template('index.html')

# # # @app.route('/search', methods=['POST'])
# # # def search():
# # #     """API endpoint for searching videos"""
# # #     data = request.get_json()
# # #     query = data.get('query', '').strip()
    
# # #     if not query:
# # #         return jsonify({'error': 'Please enter a query'}), 400
    
# # #     # Determine if this is a compound phrase
# # #     is_compound = ',' in query or len(query.split()) > 3
# # #     k = 3 if is_compound else 1  # Return multiple videos for compound phrases
    
# # #     results = get_video_playlist(query, k=k)
# # #     return jsonify({'results': results, 'is_compound': is_compound})

# # # @app.route('/videos/<filename>')
# # # def serve_video(filename):
# # #     """Serve video files from the knowledge_base/videos directory"""
# # #     return send_from_directory('knowledge_base/videos', filename)

# # # @app.route('/health')
# # # def health():
# # #     """Health check endpoint"""
# # #     return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# # # if __name__ == '__main__':
# # #     print("Loading ISL RAG Translator Demo...")
    
# # #     # Check if the database exists
# # #     if not os.path.exists('video_index.faiss') or not os.path.exists('index_map.json'):
# # #         print("❌ Database not found. Please run 'python setup_database.py' first.")
# # #         exit(1)
    
# # #     # Load components
# # #     if load_components():
# # #         print("✅ Components loaded successfully!")
# # #         print("🌐 Starting web server...")
# # #         print("📱 Open your browser and go to: http://localhost:5000")
# # #         app.run(debug=True, host='0.0.0.0', port=5000)
# # #     else:
# # #         print("❌ Failed to load components. Please check your setup.")
# # #         exit(1)

# # # app.py
# # import os
# # import re
# # import json
# # import logging
# # import string
# # from typing import Optional, Tuple, List, Dict

# # import numpy as np
# # import faiss
# # from sentence_transformers import SentenceTransformer
# # from flask import Flask, render_template, jsonify, request, send_from_directory

# # # ---------------------------
# # # Config / Tuning parameters
# # # ---------------------------
# # MAX_NGRAM = 4  # max number of words to consider as a chunk when searching (tuneable)
# # # Distance thresholds (L2 distances returned by faiss.search). Tune as needed.
# # # Smaller = stricter match. These are conservative defaults that worked well in tests.
# # DIST_THRESHOLDS = {
# #     1: 1.1,   # single word (allow more leniency to capture typos like "hii" -> "hi")
# #     2: 0.9,   # two-word chunk like "whats up"
# #     3: 0.75,  # three+ word chunk
# #     4: 0.6,
# # }
# # FALLBACK_FINGERSPELL = True  # last resort when nothing else matches
# # MODEL_NAME = "all-MiniLM-L6-v2"  # same model you used to build the index

# # # ---------------------------
# # # Setup logging + Flask app
# # # ---------------------------
# # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# # log = logging.getLogger("isl_rag")
# # app = Flask(__name__)

# # # ---------------------------
# # # Globals loaded at start
# # # ---------------------------
# # model: Optional[SentenceTransformer] = None
# # index: Optional[faiss.Index] = None
# # index_to_filename: Dict[int, str] = {}
# # metadata_list: List[Dict] = []
# # text_to_file_map: Dict[str, str] = {}
# # file_to_text_map: Dict[str, str] = {}
# # known_phrases_sorted: List[str] = []
# # max_known_ngram = 1

# # # ---------------------------
# # # Normalization helpers
# # # ---------------------------
# # def normalize_text(s: str) -> str:
# #     """Lowercase + remove punctuation (including apostrophes) and collapse whitespace."""
# #     if s is None:
# #         return ""
# #     s = s.lower().strip()
# #     # remove punctuation except keep letters/numbers/spaces
# #     s = re.sub(r"[^\w\s]", "", s)
# #     s = re.sub(r"\s+", " ", s)
# #     return s

# # def dedupe_letters(s: str) -> str:
# #     """Collapse repeated consecutive letters (hii -> hi). Useful for typos like 'hii'."""
# #     return re.sub(r"(.)\1{1,}", r"\1", s)

# # # ---------------------------
# # # Load model/index/metadata
# # # ---------------------------
# # def load_components():
# #     global model, index, index_to_filename, metadata_list, text_to_file_map, \
# #            file_to_text_map, known_phrases_sorted, max_known_ngram

# #     log.info("Loading model: %s", MODEL_NAME)
# #     model = SentenceTransformer(MODEL_NAME)

# #     index_path = "video_index.faiss"
# #     map_path = "index_map.json"
# #     meta_path = "knowledge_base/metadata.json"

# #     if not os.path.exists(index_path) or not os.path.exists(map_path) or not os.path.exists(meta_path):
# #         log.error("Missing one of required files: %s, %s, %s", index_path, map_path, meta_path)
# #         raise FileNotFoundError("Please run setup_database.py and ensure index_map.json + video_index.faiss + metadata.json exist.")

# #     log.info("Loading FAISS index from %s", index_path)
# #     index = faiss.read_index(index_path)

# #     with open(map_path, "r", encoding="utf-8") as f:
# #         raw_map = json.load(f)
# #         # ensure int keys
# #         index_to_filename = {int(k): v for k, v in raw_map.items()}

# #     with open(meta_path, "r", encoding="utf-8") as f:
# #         metadata_list = json.load(f)

# #     # Build maps (several normalized forms to increase robust matching)
# #     text_to_file_map = {}
# #     file_to_text_map = {}
# #     for item in metadata_list:
# #         text = (item.get("text") or "").strip()
# #         filename = item.get("file")
# #         if not text or not filename:
# #             continue

# #         file_to_text_map[filename] = text

# #         # raw lower form
# #         text_l = text.lower()
# #         text_to_file_map.setdefault(text_l, filename)

# #         # normalized (remove punctuation / apostrophes) -> "what's up" => "whats up"
# #         n = normalize_text(text_l)
# #         if n:
# #             text_to_file_map.setdefault(n, filename)

# #         # de-duped letters variant for resilience
# #         dedup = dedupe_letters(n)
# #         if dedup:
# #             text_to_file_map.setdefault(dedup, filename)

# #     # Prepare known phrases sorted: prefer more words first, then longer length
# #     known_phrases_sorted = sorted(
# #         set(text_to_file_map.keys()),
# #         key=lambda s: (-len(s.split()), -len(s))
# #     )
# #     if known_phrases_sorted:
# #         max_known_ngram = min(MAX_NGRAM, max(len(p.split()) for p in known_phrases_sorted))
# #     else:
# #         max_known_ngram = 1

# #     log.info("Loaded %d metadata entries. Known phrase max ngram=%d", len(text_to_file_map), max_known_ngram)


# # # ---------------------------
# # # Semantic search helper
# # # ---------------------------
# # def semantic_search_top(query: str) -> Optional[Tuple[str, str, float]]:
# #     """
# #     Encode query and return top matching (filename, matched_text, distance).
# #     Returns None if index/model not available or mapping fails.
# #     """
# #     if index is None or model is None:
# #         return None

# #     vec = model.encode([query], convert_to_numpy=True)
# #     # ensure shape (1, dim)
# #     try:
# #         distances, indices = index.search(vec, 1)
# #     except Exception as e:
# #         log.exception("FAISS search failed: %s", e)
# #         return None

# #     if indices is None or indices.size == 0:
# #         return None

# #     idx = int(indices[0][0])
# #     if idx < 0:
# #         return None

# #     dist = float(distances[0][0])
# #     filename = index_to_filename.get(idx)
# #     if not filename:
# #         return None

# #     matched_text = file_to_text_map.get(filename, "")
# #     return (filename, matched_text, dist)

# # # ---------------------------
# # # Main search algorithm
# # # ---------------------------
# # def get_video_playlist(query_text: str) -> List[Dict]:
# #     """
# #     Greedy multi-tier search:
# #     1) Try the longest exact phrase at the start (using normalized forms)
# #     2) Try semantic match on n-grams (longer first)
# #     3) Try de-duped-word semantic match (handles 'hii')
# #     4) Last-resort: fingerspell characters (only if enabled)
# #     """
# #     if not query_text:
# #         return []

# #     # Normalize incoming query (remove punctuation so "what's" -> "whats")
# #     q_norm = normalize_text(query_text)
# #     playlist: List[Dict] = []
# #     used_files = set()

# #     # Safety: avoid infinite loop
# #     max_iterations = 200
# #     iter_count = 0

# #     while q_norm and iter_count < max_iterations:
# #         iter_count += 1
# #         q_norm = q_norm.strip()
# #         if not q_norm:
# #             break

# #         words = q_norm.split()
# #         matched = False

# #         # Determine ngram limit for this loop
# #         ngram_limit = min(len(words), max_known_ngram, MAX_NGRAM)

# #         # 1) try exact known phrase (longest first)
# #         for n in range(ngram_limit, 0, -1):
# #             chunk = " ".join(words[:n])
# #             if chunk in text_to_file_map:
# #                 filename = text_to_file_map[chunk]
# #                 if filename not in used_files:
# #                     playlist.append({"file": filename, "text": file_to_text_map.get(filename, chunk), "score": 0.0, "match": "exact"})
# #                     used_files.add(filename)
# #                 q_norm = " ".join(words[n:]).strip()
# #                 matched = True
# #                 log.info("Exact match: '%s' -> %s", chunk, filename)
# #                 break
# #         if matched:
# #             continue

# #         # 2) semantic search on multi-word chunks (try longer first)
# #         for n in range(ngram_limit, 0, -1):
# #             chunk = " ".join(words[:n])
# #             sem = semantic_search_top(chunk)
# #             if sem:
# #                 filename, matched_text, dist = sem
# #                 threshold = DIST_THRESHOLDS.get(n, DIST_THRESHOLDS.get(1, 1.0))
# #                 if dist <= threshold:
# #                     if filename not in used_files:
# #                         playlist.append({"file": filename, "text": matched_text, "score": float(dist), "match": "semantic"})
# #                         used_files.add(filename)
# #                     q_norm = " ".join(words[n:]).strip()
# #                     matched = True
# #                     log.info("Semantic match (n=%d, dist=%.4f): '%s' -> %s (meta='%s')", n, dist, chunk, filename, matched_text)
# #                     break
# #                 else:
# #                     log.debug("Semantic candidate too far (n=%d, dist=%.4f, thresh=%.4f): '%s'", n, dist, threshold, chunk)
# #         if matched:
# #             continue

# #         # 3) try de-duplicated first word (handles 'hii' -> 'hi')
# #         first_word = words[0]
# #         dedup = dedupe_letters(first_word)
# #         if dedup != first_word:
# #             sem = semantic_search_top(dedup)
# #             if sem:
# #                 filename, matched_text, dist = sem
# #                 if dist <= DIST_THRESHOLDS.get(1, 1.1):
# #                     if filename not in used_files:
# #                         playlist.append({"file": filename, "text": matched_text, "score": float(dist), "match": "dedup-semantic"})
# #                         used_files.add(filename)
# #                     q_norm = " ".join(words[1:]).strip()
# #                     matched = True
# #                     log.info("Dedup match: '%s' -> '%s' (file=%s, dist=%.4f)", first_word, dedup, filename, dist)
# #                     continue

# #         # 4) Fallback: fingerspell single first word (rare)
# #         if FALLBACK_FINGERSPELL:
# #             # attempt to find character videos for each letter
# #             spelled = []
# #             any_char_found = False
# #             for ch in first_word:
# #                 if ch in text_to_file_map:
# #                     fname = text_to_file_map[ch]
# #                     if fname not in used_files:
# #                         playlist.append({"file": fname, "text": file_to_text_map.get(fname, ch), "score": 0.0, "match": "fingerspell"})
# #                         used_files.add(fname)
# #                     any_char_found = True
# #                     spelled.append(ch)
# #                 else:
# #                     log.debug("No char video for '%s'", ch)
# #             if any_char_found:
# #                 log.info("Fingerspelling fallback used for word '%s' -> %s", first_word, " ".join(spelled))
# #                 q_norm = " ".join(words[1:]).strip()
# #                 continue

# #         # If we reach here nothing matched and we either disabled fingerspell or couldn't find char videos.
# #         # To avoid infinite loop, consume the first word and continue (but log a warning).
# #         log.warning("No match for leading token '%s'. Consuming it to avoid stall.", words[0])
# #         q_norm = " ".join(words[1:]).strip()

# #     if iter_count >= max_iterations:
# #         log.error("Reached maximum iterations while parsing query. Partial playlist returned.")

# #     return playlist

# # # ---------------------------
# # # Flask routes
# # # ---------------------------
# # @app.route("/")
# # def index_page():
# #     return render_template("index.html")

# # @app.route("/search", methods=["POST"])
# # def search_route():
# #     payload = request.get_json(force=True, silent=True) or {}
# #     query = payload.get("query", "")
# #     if not isinstance(query, str) or not query.strip():
# #         return jsonify({"error": "Empty query"}), 400

# #     playlist = get_video_playlist(query)
# #     return jsonify({"playlist": playlist})

# # @app.route("/videos/<path:filename>")
# # def serve_video(filename):
# #     return send_from_directory("knowledge_base/videos", filename)

# # @app.route("/health")
# # def health():
# #     ready = model is not None and index is not None and bool(text_to_file_map)
# #     return jsonify({"status": "ok" if ready else "not_ready", "model_loaded": model is not None}), (200 if ready else 503)

# # # ---------------------------
# # # Start server
# # # ---------------------------
# # if __name__ == "__main__":
# #     load_components()
# #     log.info("Starting Flask server on 0.0.0.0:5000")
# #     app.run(host="0.0.0.0", port=5000, debug=True)


# import os
# import json
# import numpy as np
# import faiss
# import string # Import the string library for punctuation
# from sentence_transformers import SentenceTransformer
# from flask import Flask, render_template, jsonify, request, send_from_directory

# # --- 1. Initialize the Flask App ---
# app = Flask(__name__)

# # --- Global variables to hold the loaded components ---
# model = None
# index = None
# index_to_filename = None
# metadata_list = None
# known_phrases_sorted = None
# text_to_file_map = None

# # --- 2. Load AI Components ---
# def load_components():
#     """
#     Loads the AI model, FAISS index, and all necessary mappings into memory.
#     """
#     global model, index, index_to_filename, metadata_list, known_phrases_sorted, text_to_file_map
    
#     print("Loading AI components...")
#     model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     index_path = 'video_index.faiss'
#     if os.path.exists(index_path):
#         index = faiss.read_index(index_path)
#     else:
#         raise FileNotFoundError(f"FAISS index file not found at {index_path}. Please run setup_database.py first.")

#     map_path = 'index_map.json'
#     if os.path.exists(map_path):
#         with open(map_path, 'r') as f:
#             index_to_filename = {int(k): v for k, v in json.load(f).items()}
#     else:
#         raise FileNotFoundError(f"Index map file not found at {map_path}. Please run setup_database.py first.")
        
#     meta_path = 'knowledge_base/metadata.json'
#     if os.path.exists(meta_path):
#          with open(meta_path, 'r') as f:
#             metadata_list = json.load(f)
#     else:
#         raise FileNotFoundError(f"Metadata file not found at {meta_path}.")

#     # Create helper mappings for the new logic
#     known_phrases_sorted = sorted([item['text'].lower() for item in metadata_list], key=len, reverse=True)
#     text_to_file_map = {item['text'].lower(): item['file'] for item in metadata_list}

#     print("✅ AI components loaded successfully.")

# # --- 3. The Final "Three-Tiered" Search Logic ---
# def get_video_playlist(query_text):
#     """
#     Final Logic: A three-tiered approach for robust translation.
#     1. Try to match the longest exact phrase.
#     2. If no phrase, use semantic search for the closest single word.
#     3. If no similar word, fall back to fingerspelling the unknown word.
#     """
#     print(f"\nPerforming three-tiered search for query: '{query_text}'")
    
#     playlist = []
    
#     # CORRECTED PUNCTUATION HANDLING: Replace all punctuation with spaces
#     # This turns "hi,how are you" into "hi how are you"
#     space_maker = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
#     remaining_query = query_text.lower().translate(space_maker).strip()

#     while len(remaining_query) > 0:
#         found_match_phrase = None
        
#         # --- Tier 1: Try to find a long, exact phrase match ---
#         for phrase in known_phrases_sorted:
#             if remaining_query.startswith(phrase):
#                 # Check for word boundary to avoid matching "art" in "start"
#                 if len(remaining_query) == len(phrase) or remaining_query[len(phrase)] == ' ':
#                     found_match_phrase = phrase
#                     break
        
#         if found_match_phrase:
#             print(f"  - Tier 1 Match: Found exact phrase '{found_match_phrase}'")
#             filename = text_to_file_map[found_match_phrase]
#             playlist.append({"file": filename, "text": found_match_phrase, "score": 0.0})
#             remaining_query = remaining_query[len(found_match_phrase):].strip()
        
#         else:
#             words = remaining_query.split()
#             if not words: break
            
#             first_word = words[0]
            
#             # --- Tier 2: Use semantic search for the first word ---
#             word_vector = model.encode([first_word], convert_to_numpy=True)
#             distances, indices = index.search(word_vector, 1)
            
#             SIMILARITY_THRESHOLD = 0.5 
            
#             if indices.size > 0 and distances[0][0] < SIMILARITY_THRESHOLD:
#                 print(f"  - Tier 2 Match: Found semantic match for '{first_word}'")
#                 idx = indices[0][0]
#                 filename = index_to_filename.get(idx, "Unknown")
#                 matched_text = [item['text'] for item in metadata_list if item['file'] == filename][0]
#                 playlist.append({
#                     "file": filename,
#                     "text": matched_text,
#                     "score": float(distances[0][0])
#                 })
#                 remaining_query = remaining_query[len(first_word):].strip()
#             else:
#                 # --- Tier 3: Fallback to Fingerspelling ---
#                 print(f"  - Tier 3 Fallback: No close match for '{first_word}'. Fingerspelling.")
#                 for char in first_word:
#                     if char in text_to_file_map:
#                         filename = text_to_file_map[char]
#                         playlist.append({"file": filename, "text": char, "score": 0.0})
#                         print(f"    - Found character sign: '{char}'")
#                     else:
#                         print(f"    - No video found for character: '{char}'")
#                 remaining_query = remaining_query[len(first_word):].strip()

#     return playlist

# # --- 4. Define Web Routes (API Endpoints) ---

# @app.route('/')
# def index_page():
#     """Serves the main HTML page."""
#     return render_template('index.html')

# @app.route('/search', methods=['POST'])
# def search():
#     """Handles the search query from the frontend."""
#     data = request.get_json()
#     query = data.get('query', '')
#     if not query:
#         return jsonify({"error": "Empty query"}), 400
    
#     playlist = get_video_playlist(query)
    
#     return jsonify({"playlist": playlist})

# @app.route('/videos/<path:filename>')
# def serve_video(filename):
#     """Serves the video files from the knowledge_base directory."""
#     return send_from_directory('knowledge_base/videos', filename)

# @app.route('/health')
# def health_check():
#     """A simple health check endpoint to see if the server is running."""
#     return jsonify({"status": "ok"}), 200

# # --- 5. Run the Application ---
# if __name__ == '__main__':
#     load_components()
#     app.run(host='0.0.0.0', port=5000, debug=True)



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

