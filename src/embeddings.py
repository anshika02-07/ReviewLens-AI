from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(df, text_col="review_text"):
    model = load_embedding_model()

    reviews = (
        df[text_col]
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