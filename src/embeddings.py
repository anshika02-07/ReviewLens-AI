from sentence_transformers import SentenceTransformer
import streamlit as st

# =====================================================
# Load Embedding Model Only Once
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =====================================================
# Generate Embeddings
# =====================================================

def generate_embeddings(df):

    model = load_embedding_model()

    reviews = (
        df["processed_review"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(

        reviews,

        batch_size=64,

        show_progress_bar=True,

        convert_to_numpy=True

    )

    return embeddings