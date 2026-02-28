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
