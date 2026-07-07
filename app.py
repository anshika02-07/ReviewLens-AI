import streamlit as st
import pandas as pd
import plotly.express as px

from src.analysis import analyze_reviews
from src.sentiment import run_sentiment
from src.preprocess import preprocess_reviews
from src.column_mapper import detect_columns
from src.clean_data import clean_dataset

# ======================================================
# Page Config
# ======================================================

st.set_page_config(
    page_title="ReviewLens AI",
    page_icon="🤖",
    layout="wide"
)

st.title("ReviewLens AI")

st.markdown(
"""
### AI Powered Customer Review Intelligence

Analyze any review dataset using NLP, Transformers and AI.

Upload • Analyze • Visualize • Search
"""
)

st.divider()

# ======================================================
# Upload CSV
# ======================================================

uploaded_file = st.file_uploader("Upload Review Dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset Loaded Successfully")

col1, col2 = st.columns(2)
with col1:
    st.info(f"Rows : {len(df)}")
with col2:
    st.info(f"Columns : {len(df.columns)}")

# ======================================================
# Detect Columns Automatically
# ======================================================

st.header("🔍 Automatic Column Detection")

detected = detect_columns(df)

for standard_name, detected_column in detected.items():
    if detected_column is not None:
        st.success(f"✅ {standard_name} → {detected_column}")
    else:
        st.warning(f"⚠ {standard_name} not detected.")

# ======================================================
# Manual Mapping
# ======================================================

st.header("Manual Column Mapping")

final_mapping = {}
required_columns = ["review_text", "rating"]
optional_columns = [
    "review_title", "product_id", "category",
    "age", "recommended", "helpful_votes"
]

for column in required_columns:
    if detected[column] is not None:
        final_mapping[column] = detected[column]
    else:
        st.warning(f"{column} not detected.")
        final_mapping[column] = st.selectbox(
            f"Select {column}", list(df.columns), key=column
        )

for column in optional_columns:
    if detected[column] is not None:
        final_mapping[column] = detected[column]

rename_dict = {v: k for k, v in final_mapping.items()}
df = df.rename(columns=rename_dict)

st.divider()

if not st.button("Start Analysis", use_container_width=True):
    st.info("Click 'Start Analysis' to generate dashboard.")
    st.stop()

# ======================================================
# Clean Dataset
# ======================================================

df = clean_dataset(df)

if "review_text" not in df.columns:
    st.error("Review Text column is required.")
    st.stop()

st.success("Dataset Ready for Analysis")
st.divider()

# ======================================================
# Basic (fast) Dashboard — always shown, no model calls
# ======================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Reviews", len(df))
with col2:
    if "rating" in df.columns:
        st.metric("Average Rating", round(df["rating"].mean(), 2))
with col3:
    if "recommended" in df.columns:
        st.metric("Recommendation %", round(df["recommended"].mean() * 100, 2))
with col4:
    if "category" in df.columns:
        st.metric("Categories", df["category"].nunique())

st.divider()

if "rating" in df.columns:
    st.subheader("Rating Distribution")
    rating_counts = df["rating"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index, y=rating_counts.values,
        labels={"x": "Rating", "y": "Reviews"},
        title="Customer Rating Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

if "category" in df.columns:
    st.subheader("Top Product Categories")
    category_counts = df["category"].value_counts().head(10)
    fig = px.bar(
        category_counts, x=category_counts.values, y=category_counts.index,
        orientation="h", title="Top Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

if "age" in df.columns:
    st.subheader("Customer Age Distribution")
    valid_age = df[df["age"] > 0]
    fig = px.histogram(valid_age, x="age", nbins=20, title="Customer Age")
    st.plotly_chart(fig, use_container_width=True)

if "recommended" in df.columns:
    st.subheader("Recommendation Distribution")
    recommendation = df["recommended"].value_counts()
    recommendation.index = ["Recommended", "Not Recommended"]
    fig = px.pie(values=recommendation.values, names=recommendation.index, hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# Slow AI Analysis — gated behind a button + sample size
# ======================================================

st.divider()
st.header("🤖 AI-Powered Sentiment Analysis")

st.warning(
    "DistilBERT sentiment analysis is slow on large datasets. "
    "Use the sample size slider below to keep the demo fast. "
    "Full-dataset runs should use caching (see README)."
)

sample_size = st.slider(
    "Number of reviews to analyze",
    min_value=100,
    max_value=min(5000, len(df)),
    value=min(2000, len(df)),
    step=100
)

run_ai = st.button("Run AI Analysis", use_container_width=True)

if run_ai:
    with st.spinner(f"Running sentiment analysis on {sample_size} reviews..."):
        sample_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sample_df = preprocess_reviews(sample_df)
        sample_df = run_sentiment(sample_df)
        results = analyze_reviews(sample_df)

    st.success("AI Analysis Completed")

    st.subheader("Sentiment Summary")
    s = results["summary"]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Positive Reviews", s.get("Positive Reviews", "N/A"))
    with c2:
        st.metric("Negative Reviews", s.get("Negative Reviews", "N/A"))

    if results["positive_words"] is not None:
        st.subheader("Top Positive Words")
        st.dataframe(results["positive_words"], use_container_width=True)

    if results["negative_words"] is not None:
        st.subheader("Top Negative Words")
        st.dataframe(results["negative_words"], use_container_width=True)

    st.subheader("Sample Results")
    st.dataframe(
        sample_df[["review_text", "sentiment", "confidence"]].head(20),
        use_container_width=True
    )

    st.download_button(
        "Download Analyzed Sample",
        sample_df.to_csv(index=False),
        "analyzed_reviews.csv",
        "text/csv"
    )

# ======================================================
# Keyword Search (fast, no model needed)
# ======================================================

st.divider()
st.header("🔎 Keyword Search")

query = st.text_input("Search any keyword")

if query:
    result = df[df["review_text"].str.contains(query, case=False, na=False)]
    st.write(f"Found {len(result)} matching reviews")
    st.dataframe(result.head(20), use_container_width=True)

# ======================================================
# Helpful / Longest Reviews
# ======================================================

if "helpful_votes" in df.columns:
    st.header("Top Helpful Reviews")
    helpful = df.sort_values("helpful_votes", ascending=False).head(10)
    st.dataframe(helpful, use_container_width=True)

st.header("Longest Reviews")
df["review_length"] = df["review_text"].astype(str).str.split().str.len()
longest = df.sort_values("review_length", ascending=False).head(10)
st.dataframe(longest, use_container_width=True)

# ======================================================
# Full Dataset Preview + Download
# ======================================================

st.divider()
st.header("📄 Standardized Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.download_button(
    "Download Cleaned Dataset",
    df.to_csv(index=False),
    "processed_reviews.csv",
    "text/csv"
)