# """
# retriever.py — The search engine for your catalog.

# Given a query like "senior Java developer who works with stakeholders",
# this returns the most relevant assessments from your catalog.

# It also handles the test_type mapping — converting SHL's category names
# ("Knowledge & Skills") into short codes ("K") that the API needs to return.
# """

# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss

# # ── Load everything once at startup ──────────────────────────────────────────
# # Loading these takes a few seconds, so we do it once when Python imports this file
# # rather than every time a search request comes in.

# print("Loading catalog and search index...")

# with open("catalog.json", "r", encoding="utf-8") as f:
#     CATALOG = json.load(f)

# INDEX = faiss.read_index("faiss.index")
# MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# print(f"Retriever ready. {len(CATALOG)} assessments indexed.")

# # ── Test type mapping ─────────────────────────────────────────────────────────
# # The catalog uses full category names. The API spec uses short codes.
# # This dictionary maps one to the other.

# KEY_TO_TYPE = {
#     "Knowledge & Skills": "K",
#     "Personality & Behavior": "P",
#     "Ability & Aptitude": "A",
#     "Biodata & Situational Judgment": "B",
#     "Simulations": "S",
#     "Competencies": "C",
#     "Development & 360": "D",
#     "Assessment Exercises": "E",
# }

# def get_test_type(keys: list) -> str:
#     """
#     Convert a list of category keys to a comma-separated type code.
#     Example: ["Knowledge & Skills", "Simulations"] → "K,S"
#     """
#     codes = []
#     for key in keys:
#         code = KEY_TO_TYPE.get(key)
#         if code and code not in codes:
#             codes.append(code)
#     return ",".join(codes) if codes else "A"  # Default to Ability if unknown


# def format_for_api(item: dict) -> dict:
#     """
#     Convert a catalog item into the exact format the API needs to return.
#     The catalog uses 'link', but the API spec needs 'url'.
#     """
#     return {
#         "name": item.get("name", "Unknown"),
#         "url": item.get("link", ""),
#         "test_type": get_test_type(item.get("keys", []))
#     }


# def retrieve(query: str, k: int = 15) -> list:
#     """
#     Given a plain-English query, return the top-k most relevant catalog items.
    
#     We retrieve 15 by default and let the LLM pick the best 1-10.
#     This gives the LLM choices without overwhelming it.
    
#     Args:
#         query: The search query (usually the conversation context)
#         k: How many results to return
    
#     Returns:
#         List of catalog items (full dictionaries, not just formatted ones)
#     """
#     # Convert query to a vector
#     query_embedding = MODEL.encode([query], convert_to_numpy=True)
    
#     # Normalize (same as we did during indexing)
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
#     # Search the FAISS index
#     scores, indices = INDEX.search(query_embedding.astype("float32"), k)
    
#     # Return the matching catalog items
#     results = []
#     for idx in indices[0]:
#         if 0 <= idx < len(CATALOG):  # Safety check
#             results.append(CATALOG[idx])
    
#     return results


# def get_catalog_summary_for_llm(items: list) -> str:
#     """
#     Format a list of catalog items into readable text for the LLM.
#     This goes into the prompt so the LLM knows what it can recommend.
#     """
#     lines = []
#     for item in items:
#         name = item.get("name", "")
#         link = item.get("link", "")
#         desc = item.get("description", "")[:250]  # Keep descriptions concise
#         keys = ", ".join(item.get("keys", []))
#         levels = ", ".join(item.get("job_levels", []))
#         duration = item.get("duration", "Not specified")
#         remote = item.get("remote", "unknown")
#         adaptive = item.get("adaptive", "unknown")
#         langs = ", ".join(item.get("languages", [])[:6])
#         if len(item.get("languages", [])) > 6:
#             langs += f" (+{len(item.get('languages', [])) - 6} more)"
        
#         test_type = get_test_type(item.get("keys", []))
        
#         lines.append(
#             f"---\n"
#             f"Name: {name}\n"
#             f"URL: {link}\n"
#             f"Type Code: {test_type}\n"
#             f"Categories: {keys}\n"
#             f"Job Levels: {levels}\n"
#             f"Duration: {duration} | Remote: {remote} | Adaptive: {adaptive}\n"
#             f"Languages: {langs}\n"
#             f"Description: {desc}"
#         )
    
#     return "\n".join(lines)


# def search_by_names(names: list) -> list:
#     """
#     Find specific catalog items by name (for comparison requests).
#     Used when the user asks "what's the difference between X and Y?"
#     Returns catalog items that match any of the given names.
#     """
#     results = []
#     names_lower = [n.lower().strip() for n in names]
    
#     for item in CATALOG:
#         item_name_lower = item.get("name", "").lower()
#         for name in names_lower:
#             # Partial match — "OPQ" matches "Occupational Personality Questionnaire OPQ32r"
#             if name in item_name_lower or item_name_lower in name:
#                 if item not in results:
#                     results.append(item)
#                 break
    
#     return results




"""
retriever.py — Lightweight TF-IDF based retrieval.
Uses ~30MB RAM. Works well for 377 assessments.
"""

import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Load index once at startup ────────────────────────────────────────────────
print("Loading catalog and search index...")

with open("tfidf_index.pkl", "rb") as f:
    data = pickle.load(f)

VECTORIZER = data["vectorizer"]
TFIDF_MATRIX = data["matrix"]
CATALOG = data["catalog"]

print(f"Retriever ready. {len(CATALOG)} assessments indexed.")

# ── Test type mapping ─────────────────────────────────────────────────────────

KEY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

def get_test_type(keys: list) -> str:
    codes = []
    for key in keys:
        code = KEY_TO_TYPE.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "A"


def retrieve(query: str, k: int = 15) -> list:
    """
    Given a plain-English query, return the top-k most relevant assessments.
    Uses TF-IDF cosine similarity — lightweight and fast.
    """
    # Transform query to TF-IDF vector
    query_vec = VECTORIZER.transform([query.lower()])
    
    # Compute cosine similarity against all catalog items
    scores = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()
    
    # Get top-k indices sorted by score
    top_indices = np.argsort(scores)[::-1][:k]
    
    return [CATALOG[i] for i in top_indices]


def get_catalog_summary_for_llm(items: list) -> str:
    """Format retrieved items for the LLM prompt."""
    lines = []
    for item in items:
        name = item.get("name", "")
        link = item.get("link", "")
        desc = item.get("description", "")[:250]
        keys = ", ".join(item.get("keys", []))
        levels = ", ".join(item.get("job_levels", []))
        duration = item.get("duration", "Not specified")
        remote = item.get("remote", "unknown")
        adaptive = item.get("adaptive", "unknown")
        langs = ", ".join(item.get("languages", [])[:6])
        if len(item.get("languages", [])) > 6:
            langs += f" (+{len(item.get('languages', [])) - 6} more)"
        
        test_type = get_test_type(item.get("keys", []))
        
        lines.append(
            f"---\n"
            f"Name: {name}\n"
            f"URL: {link}\n"
            f"Type Code: {test_type}\n"
            f"Categories: {keys}\n"
            f"Job Levels: {levels}\n"
            f"Duration: {duration} | Remote: {remote} | Adaptive: {adaptive}\n"
            f"Languages: {langs}\n"
            f"Description: {desc}"
        )
    
    return "\n".join(lines)


def search_by_names(names: list) -> list:
    """Find specific catalog items by name (for comparison requests)."""
    results = []
    names_lower = [n.lower().strip() for n in names]
    
    for item in CATALOG:
        item_name_lower = item.get("name", "").lower()
        for name in names_lower:
            if name in item_name_lower or item_name_lower in name:
                if item not in results:
                    results.append(item)
                break
    
    return results


def format_for_api(item: dict) -> dict:
    return {
        "name": item.get("name", "Unknown"),
        "url": item.get("link", ""),
        "test_type": get_test_type(item.get("keys", []))
    }