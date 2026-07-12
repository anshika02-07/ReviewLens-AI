import streamlit as st
import pandas as pd
import plotly.express as px

from src.analysis import analyze_reviews
from src.sentiment import run_sentiment
from src.preprocess import preprocess_reviews
from src.column_mapper import detect_columns
from src.clean_data import clean_dataset
from src.cache_utils import get_file_hash, save_cache, load_cache
from src.embeddings import generate_embeddings, load_embedding_model
from src.faiss_index import build_index
from src.semantic_search import semantic_search
from src.rag import generate_answer
from src.clustering import cluster_reviews
from src.executive_summary import generate_executive_summary

# ======================================================
# Page Config
# ======================================================

st.set_page_config(page_title="ReviewLens AI", page_icon="🤖", layout="wide")

st.title("ReviewLens AI")
st.markdown(
    "### AI-Powered Customer Review Intelligence\n"
    "Upload any review dataset — every tab below works independently; "
    "nothing needs to be run 'first' in another tab."
)

with st.expander("ℹ️ How the RAG chatbot works (click to learn)"):
    st.markdown("""
    1. **Embed**: every review is converted into a vector (a list of numbers
       capturing its meaning) using a Sentence-Transformer model.
    2. **Index**: those vectors are stored in a FAISS index for fast similarity lookup.
    3. **Retrieve**: when you ask a question, it's embedded the same way, and
       FAISS finds the 8 most similar reviews.
    4. **Augment**: those 8 reviews are inserted into the prompt as context.
    5. **Generate**: Gemini answers using *only* that context — not general
       knowledge — which is what makes the answer traceable back to real data.

    The index (steps 1-2) is built automatically the first time you use
    Semantic Search or Chatbot — you don't need to trigger it manually.
    """)

st.divider()

# ======================================================
# Display helper
# ======================================================

def get_display_df(df, extra_cols=None, truncate_len=150):
    base_cols = ["review_text", "review_title", "rating", "category"]
    if extra_cols:
        base_cols += extra_cols
    cols = [c for c in base_cols if c in df.columns]
    display = df[cols].copy()
    if "review_text" in display.columns:
        display["review_text"] = display["review_text"].astype(str).str.slice(0, truncate_len) + "..."
    return display
def show_chart(fig, min_points=2):
    """
    Wraps st.plotly_chart with consistent styling and automatically
    figures out how many data points the chart has — no need to pass
    that in manually for every single chart.
    """
    total_points = 0
    for trace in fig.data:
        if getattr(trace, "x", None) is not None:
            total_points += len(trace.x)
        elif getattr(trace, "y", None) is not None:
            total_points += len(trace.y)

    if total_points < min_points:
        st.info("Not enough data in this dataset to show this chart.")
        return

    fig.update_layout(
        template="plotly_dark",
        font=dict(size=13),
        margin=dict(l=40, r=20, t=50, b=40),
        title_font_size=18,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# USAGE — replace calls like this:
#
#   fig = px.bar(x=rating_counts.index, y=rating_counts.values, ...)
#   show_chart(fig)
#
# with this:
#
#   fig = px.bar(x=rating_counts.index, y=rating_counts.values, ...)
#   show_chart(fig, data_len=len(rating_counts))
# ------------------------------------------------------------

# ======================================================
# Self-sufficient builders — each tab calls these directly.
# Nothing requires visiting another tab first. Results are
# cached both in session_state (survives reruns) AND on disk
# via file_hash (survives even closing/reopening the app).
# ======================================================

def get_or_build_index(df, file_hash):
    if (st.session_state.get("faiss_index") is not None
            and st.session_state.get("indexed_file_hash") == file_hash):
        return st.session_state.embed_model, st.session_state.faiss_index, st.session_state.search_df

    with st.spinner("Preparing semantic index (embeddings + FAISS) — runs once per file..."):
        embeddings = load_cache(file_hash, "embeddings_raw")
        if embeddings is None:
            embeddings = generate_embeddings(df, text_col="review_text")
            save_cache(file_hash, "embeddings_raw", embeddings)

        model = load_embedding_model()
        index = build_index(embeddings)

        st.session_state.embed_model = model
        st.session_state.faiss_index = index
        st.session_state.search_df = df
        st.session_state.indexed_file_hash = file_hash

    return model, index, df


def get_or_run_sentiment(df, sample_size, file_hash):
    if (st.session_state.get("ai_results") is not None
            and st.session_state.get("sentiment_file_hash") == file_hash
            and st.session_state.get("sentiment_sample_size") == sample_size):
        return st.session_state.sample_df, st.session_state.ai_results

    cache_key = f"sentiment_{sample_size}"
    cached = load_cache(file_hash, cache_key)

    if cached is not None:
        sample_df, results = cached
    else:
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        sample_df = preprocess_reviews(sample_df)

        progress_bar = st.progress(0, text="Starting sentiment analysis...")

        def update_progress(fraction):
            progress_bar.progress(fraction, text=f"Analyzing reviews... {int(fraction*100)}%")

        sample_df = run_sentiment(sample_df, progress_callback=update_progress)
        progress_bar.empty()

        results = analyze_reviews(sample_df)
        save_cache(file_hash, cache_key, (sample_df, results))

    st.session_state.sample_df = sample_df
    st.session_state.ai_results = results
    st.session_state.sentiment_file_hash = file_hash
    st.session_state.sentiment_sample_size = sample_size

    return sample_df, results


# ======================================================
# Upload CSV
# ======================================================

uploaded_file = st.file_uploader("Upload Review Dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

file_hash = get_file_hash(uploaded_file)

if st.session_state.get("current_file_hash") != file_hash:
    for key in ["analysis_started", "ai_results", "sample_df", "chat_history",
                "faiss_index", "embed_model", "search_df", "indexed_file_hash",
                "cluster_df", "cluster_summary", "exec_summary",
                "sentiment_file_hash", "sentiment_sample_size"]:
        st.session_state[key] = None
    st.session_state.current_file_hash = file_hash
    st.session_state.analysis_started = False
    st.session_state.chat_history = []

df = pd.read_csv(uploaded_file)
st.success("Dataset Loaded Successfully")

col1, col2 = st.columns(2)
with col1:
    st.info(f"Rows : {len(df)}")
with col2:
    st.info(f"Columns : {len(df.columns)}")

# ======================================================
# Column Detection & Mapping
# ======================================================

st.header("🔍 Automatic Column Detection")
detected = detect_columns(df)


found_cols = {k: v for k, v in detected.items() if v is not None}
missing_cols = [k for k, v in detected.items() if v is None]
 
col_found, col_missing = st.columns(2)
 
with col_found:
    st.markdown(f"**✅ Detected ({len(found_cols)}/{len(detected)})**")
    for standard_name, detected_column in found_cols.items():
        st.markdown(f"`{standard_name}` → *{detected_column}*")
 
with col_missing:
    if missing_cols:
        st.markdown(f"**⚠ Not detected ({len(missing_cols)})**")
        for standard_name in missing_cols:
            required = " (required — pick below)" if standard_name in ["review_text", "rating"] else " (optional)"
            st.markdown(f"`{standard_name}`{required}")
    else:
        st.markdown("**All fields detected**🎉")
st.header("Manual Column Mapping")

final_mapping = {}
required_columns = ["review_text", "rating"]
optional_columns = ["review_title", "product_id", "category", "age", "recommended", "helpful_votes", "date"]

for column in required_columns:
    if detected[column] is not None:
        final_mapping[column] = detected[column]
    else:
        st.warning(f"{column} not detected.")
        final_mapping[column] = st.selectbox(
            f"Select {column}", list(df.columns), key=f"{column}_{file_hash}"
        )

for column in optional_columns:
    if detected[column] is not None:
        final_mapping[column] = detected[column]

rename_dict = {v: k for k, v in final_mapping.items()}
df = df.rename(columns=rename_dict)

standard_cols = [c for c in
    ["review_text", "review_title", "rating", "category", "age",
     "recommended", "helpful_votes", "product_id", "date"]
    if c in df.columns]
df = df[standard_cols]

st.divider()

if st.button("Start Analysis", use_container_width=True):
    st.session_state.analysis_started = True

if not st.session_state.get("analysis_started", False):
    st.info("Click 'Start Analysis' to generate dashboard.")
    st.stop()

df = clean_dataset(df)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "review_text" not in df.columns:
    st.error("Review Text column is required.")
    st.stop()

st.success("Dataset Ready for Analysis")
st.divider()

# ======================================================
# Tabs — every tab below is now self-sufficient
# ======================================================

tab_dashboard, tab_sentiment, tab_cluster, tab_summary, tab_search, tab_chat = st.tabs(
    ["📊 Dashboard", "🤖 Sentiment (BERT)", "🧩 Clustering (K-Means)",
     "📋 Executive Summary", "🔎 Semantic Search", "💬 Chatbot (RAG)"]
)

# --------------------------------------------------
# TAB 1 — Dashboard
# --------------------------------------------------
with tab_dashboard:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        st.metric("Average Rating", round(df["rating"].mean(), 2))
    with col3:
        st.metric("Recommendation %", round(df["recommended"].mean() * 100, 2))
    with col4:
        st.metric("Categories", df["category"].nunique())

    st.divider()
    st.subheader("Rating Distribution")
    rating_counts = df["rating"].value_counts().sort_index()
    fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                 labels={"x": "Rating", "y": "Reviews"}, title="Customer Rating Distribution")
    show_chart(fig)

    has_trend_data = False
    if "date" in df.columns and df["date"].notna().sum() > 5:
        st.subheader("Rating Trend Over Time")
        trend_df = df.dropna(subset=["date"]).set_index("date").resample("ME")["rating"].mean().reset_index()
        fig = px.line(trend_df, x="date", y="rating", title="Average Rating Over Time", markers=True)
        show_chart(fig)
        has_trend_data = True
    st.session_state.has_trend_data = has_trend_data

    if df["category"].nunique() > 1:
        st.subheader("Top Product Categories")
        category_counts = df["category"].value_counts().head(10)
        fig = px.bar(category_counts, x=category_counts.values, y=category_counts.index,
                     orientation="h", title="Top Categories")
        show_chart(fig)

    st.divider()
    st.subheader("🔎 Keyword Search")
    query = st.text_input("Search any keyword", key="keyword_search_box")
    if query:
        result = df[df["review_text"].str.contains(query, case=False, na=False)]
        st.write(f"Found {len(result)} matching reviews")
        st.dataframe(get_display_df(result.head(20)), use_container_width=True)

    st.download_button("Download Cleaned Dataset", df.to_csv(index=False),
                        "processed_reviews.csv", "text/csv")

# --------------------------------------------------
# TAB 2 — Sentiment Analysis
# --------------------------------------------------
with tab_sentiment:
    st.header("DistilBERT Sentiment Analysis")

    sample_size = st.slider("Number of reviews to analyze", min_value=100,
                             max_value=min(5000, len(df)), value=min(1500, len(df)), step=100)

    if st.button("Run AI Analysis", use_container_width=True):
        get_or_run_sentiment(df, sample_size, file_hash)
        st.success("AI Analysis Completed")

    if st.session_state.get("ai_results"):
        results = st.session_state.ai_results
        sample_df = st.session_state.sample_df

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
        st.dataframe(get_display_df(sample_df, extra_cols=["sentiment", "confidence"]).head(20),
                     use_container_width=True)
        st.download_button("Download Analyzed Sample", sample_df.to_csv(index=False),
                            "analyzed_reviews.csv", "text/csv")

# --------------------------------------------------
# TAB 3 — Clustering (auto-runs sentiment if needed)
# --------------------------------------------------
with tab_cluster:
    st.header("Complaint & Theme Clustering")
    st.caption("Groups reviews into themes using K-Means on embeddings, labeled by top terms per cluster.")

    n_clusters = st.slider("Number of clusters", 2, 10, 5)
    cluster_sample_size = st.slider("Sample size for clustering", 100, min(3000, len(df)), min(1000, len(df)), step=100)

    if st.button("Run Clustering", use_container_width=True):
        with st.spinner("Preparing data and clustering..."):
            sample_df, _ = get_or_run_sentiment(df, cluster_sample_size, file_hash)
            cluster_embeddings = generate_embeddings(sample_df, text_col="review_text")
            cluster_df, cluster_summary = cluster_reviews(sample_df, cluster_embeddings, n_clusters=n_clusters)
            st.session_state.cluster_df = cluster_df
            st.session_state.cluster_summary = cluster_summary
        st.success("Clustering complete")

    if st.session_state.get("cluster_summary") is not None:
        st.subheader("Cluster Summary")
        st.dataframe(st.session_state.cluster_summary, use_container_width=True)

        fig = px.bar(st.session_state.cluster_summary.reset_index(), x="cluster_label", y="Reviews",
                     title="Reviews per Cluster")
        show_chart(fig)

        st.subheader("Explore a Cluster")
        chosen = st.selectbox("Select a cluster", st.session_state.cluster_summary.index.tolist())
        examples = st.session_state.cluster_df[st.session_state.cluster_df["cluster_label"] == chosen]
        st.dataframe(get_display_df(examples, extra_cols=["sentiment"]).head(15), use_container_width=True)

# --------------------------------------------------
# TAB 4 — Executive Summary (auto-runs sentiment if needed)
# --------------------------------------------------
with tab_summary:
    st.header("Executive Summary")
    st.caption("LLM-generated business summary, grounded in sentiment results and complaint clusters.")

    summary_sample_size = st.slider("Sample size for summary", 100, min(3000, len(df)), min(1000, len(df)), step=100, key="summary_sample")

    if st.button("Generate Executive Summary", use_container_width=True):
        with st.spinner("Analyzing reviews and generating summary..."):
            sample_df, results = get_or_run_sentiment(df, summary_sample_size, file_hash)
            summary_stats = results["summary"]

            top_pos = (sample_df[sample_df["sentiment"] == "POSITIVE"]
                       .sort_values("confidence", ascending=False)["review_text"].head(6).tolist())
            top_neg = (sample_df[sample_df["sentiment"] == "NEGATIVE"]
                       .sort_values("confidence", ascending=False)["review_text"].head(6).tolist())

            trend_description = None
            if st.session_state.get("has_trend_data") and "date" in df.columns:
                trend_df = df.dropna(subset=["date"]).set_index("date").resample("ME")["rating"].mean()
                if len(trend_df) >= 2:
                    change = trend_df.iloc[-1] - trend_df.iloc[0]
                    direction = "increased" if change > 0 else "decreased"
                    trend_description = (
                        f"Average rating {direction} from {trend_df.iloc[0]:.2f} to "
                        f"{trend_df.iloc[-1]:.2f} across the observed time period."
                    )

            summary_text = generate_executive_summary(
                summary_stats, top_pos, top_neg,
                cluster_summary=st.session_state.get("cluster_summary"),
                trend_description=trend_description
            )
            st.session_state.exec_summary = summary_text

    if st.session_state.get("exec_summary"):
        st.markdown(st.session_state.exec_summary)
        st.download_button("Download Summary", st.session_state.exec_summary,
                            "executive_summary.txt", "text/plain")

# --------------------------------------------------
# TAB 5 — Semantic Search (auto-builds index)
# --------------------------------------------------
with tab_search:
    st.header("Semantic Search")
    st.caption("Search by meaning, not just exact keywords.")

    search_query = st.text_input("Search reviews by meaning", key="semantic_query")
    top_k = st.slider("Number of results", 3, 20, 8, key="semantic_top_k")

    if search_query:
        model, index, search_df = get_or_build_index(df, file_hash)
        results = semantic_search(search_query, search_df, model, index, top_k=top_k)
        st.write(f"Top {len(results)} semantically similar reviews:")
        st.dataframe(get_display_df(results, extra_cols=["Similarity"]), use_container_width=True)
    else:
        st.info("Type a query above — the semantic index builds automatically on first search.")

# --------------------------------------------------
# TAB 6 — RAG Chatbot (auto-builds index)
# --------------------------------------------------
with tab_chat:
    st.header("Chat With Your Reviews")
    st.caption("Ask a question — the semantic index builds automatically if it isn't ready yet.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask a question about the reviews...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant reviews and thinking..."):
                model, index, search_df = get_or_build_index(df, file_hash)
                retrieved = semantic_search(user_question, search_df, model, index, top_k=8)
                answer = generate_answer(user_question, retrieved,
                                          chat_history=st.session_state.chat_history[:-1])
                st.markdown(answer)

                with st.expander("📄 Reviews used to answer this"):
                    st.dataframe(get_display_df(retrieved, extra_cols=["Similarity"]), use_container_width=True)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})