"""
ReviewLens AI
Semantic Search — embeds a natural-language query and retrieves
the most similar reviews using the FAISS index.
"""

import numpy as np
from src.faiss_index import search_reviews


def embed_query(query, model):
    """
    Encode a single query string into the same embedding space
    as the review corpus.
    """
    return model.encode([query], convert_to_numpy=True)[0]


def semantic_search(query, df, model, index, top_k=8):
    """
    Returns the top_k reviews most semantically similar to `query`.
    Lower 'Similarity' (L2 distance) = more similar.
    """
    query_embedding = embed_query(query, model)
    results = search_reviews(query_embedding, index, df, top_k=top_k)
    return results.sort_values("Similarity", ascending=True)