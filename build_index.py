"""
build_index.py — Run this ONCE to build the FAISS search index.
It reads catalog.json, converts each assessment into a vector (list of numbers
that captures its meaning), and saves the index to disk.

Think of it like building the index at the back of a textbook — you do it once
so future lookups are instant.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

print("Loading catalog...")
with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

print(f"Found {len(catalog)} assessments.")

# Build a rich text description for each assessment
# The richer the text, the better the search results
# We combine name + description + job levels + keys
def build_text(item):
    name = item.get("name", "")
    desc = item.get("description", "")
    keys = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    duration = item.get("duration", "")
    remote = item.get("remote", "")
    adaptive = item.get("adaptive", "")
    langs = ", ".join(item.get("languages", [])[:5])  # First 5 languages only
    
    return (
        f"Assessment: {name}. "
        f"Description: {desc}. "
        f"Categories: {keys}. "
        f"Suitable for: {levels}. "
        f"Duration: {duration}. "
        f"Remote: {remote}. Adaptive: {adaptive}. "
        f"Languages: {langs}."
    )

texts = [build_text(item) for item in catalog]

print("Building embeddings (this takes 2-5 minutes for large catalogs)...")
# all-MiniLM-L6-v2 is small (80MB), fast, and works very well for this task
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Normalize embeddings so we get cosine similarity (measures direction, not magnitude)
# This gives better results for semantic search
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / norms

print("Building FAISS index...")
dimension = embeddings.shape[1]  # Should be 384 for this model
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (works like cosine when normalized)
index.add(embeddings.astype("float32"))

# Save both the index and the texts used (for debugging)
faiss.write_index(index, "faiss.index")
np.save("embeddings.npy", embeddings)

print(f"Done! Index saved with {index.ntotal} vectors.")
print("You can now run: uvicorn main:app --reload")