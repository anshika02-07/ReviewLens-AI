import faiss
import numpy as np
import streamlit as st

# =====================================================
# Build FAISS Index
# =====================================================

@st.cache_resource
def build_index(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# =====================================================
# Semantic Search
# =====================================================

def search_reviews(

    query_embedding,

    index,

    df,

    top_k=5

):

    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = df.iloc[
        indices[0]
    ].copy()

    results["Similarity"] = distances[0]

    return results