# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# print("Initializing sentence-transformer model...")
# # Using a lightweight, high-performance model suitable for CPU
# model = SentenceTransformer('all-MiniLM-L6-v2')

# print("Loading video metadata from knowledge_base/metadata.json...")
# with open('knowledge_base/metadata.json', 'r') as f:
#     metadata = json.load(f)

# # Extracting text descriptions to be encoded
# texts = [item['text'] for item in metadata]
# filenames = [item['file'] for item in metadata]

# print(f"Found {len(texts)} text phrases to encode.")
# print("Encoding text phrases into vectors... (This may take a moment)")
# embeddings = model.encode(texts, convert_to_numpy=True)

# print(f"Embeddings created with shape: {embeddings.shape}")
# embedding_dimension = embeddings.shape[1]

# # Setting up the FAISS index for fast similarity search
# print("Creating FAISS index...")
# # Using a simple L2 distance index
# index = faiss.IndexFlatL2(embedding_dimension) 
# # To keep track of original IDs, we wrap it with IndexIDMap
# index = faiss.IndexIDMap(index)

# # Adding the embeddings to the index with their original positions as IDs
# index.add_with_ids(embeddings, np.arange(len(texts)))

# print(f"Total vectors in index: {index.ntotal}")

# # Saving the index and the mapping to disk
# print("Saving FAISS index to 'video_index.faiss'...")
# faiss.write_index(index, 'video_index.faiss')

# # Save the mapping from the index ID to the corresponding filename
# index_to_filename = {i: filename for i, filename in enumerate(filenames)}
# print("Saving index-to-filename mapping to 'index_map.json'...")
# with open('index_map.json', 'w') as f:
#     json.dump(index_to_filename, f)

# print("\n✅ Setup complete! Your vector database is ready.")


# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# from app import normalize_text, dedupe_letters  # reuse same normalization

# # Load metadata
# with open("knowledge_base/metadata.json", "r", encoding="utf-8") as f:
#     metadata = json.load(f)

# # Normalize and dedupe text before encoding
# texts = [normalize_text(dedupe_letters(item["text"])) for item in metadata]

# # Encode
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# # Build FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# faiss.write_index(index, "knowledge_base/faiss.index")

# print(f"✅ Indexed {len(metadata)} items into FAISS (dimension={dimension}).")


import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Initializing sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading video metadata from knowledge_base/metadata.json...")
with open('knowledge_base/metadata.json', 'r') as f:
    metadata = json.load(f)

texts = [item['text'] for item in metadata]
filenames = [item['file'] for item in metadata]

print(f"Found {len(texts)} text phrases to encode.")
print("Encoding text phrases into vectors... (This may take a moment)")
embeddings = model.encode(texts, convert_to_numpy=True)

print(f"Embeddings created with shape: {embeddings.shape}")
embedding_dimension = embeddings.shape[1]

print("Creating FAISS index...")
index = faiss.IndexFlatL2(embedding_dimension) 
index = faiss.IndexIDMap(index)

index.add_with_ids(embeddings, np.arange(len(texts)))

print(f"Total vectors in index: {index.ntotal}")

print("Saving FAISS index to 'video_index.faiss'...")
faiss.write_index(index, 'video_index.faiss')

index_to_filename = {i: filename for i, filename in enumerate(filenames)}
print("Saving index-to-filename mapping to 'index_map.json'...")
with open('index_map.json', 'w') as f:
    json.dump(index_to_filename, f)

print("\n✅ Setup complete! Your vector database is ready.")
