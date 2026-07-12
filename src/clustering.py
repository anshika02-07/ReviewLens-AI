"""
ReviewLens AI
Complaint / Theme Clustering

Groups reviews into thematic clusters using K-Means on their
sentence embeddings (the same vectors used for semantic search),
then labels each cluster using its top TF-IDF terms so the output
is human-readable (e.g. "sizing, tight, small" instead of "Cluster 2").

This is the piece that demonstrates unsupervised ML + vector
representations end-to-end: embed -> cluster -> interpret.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_reviews(df, embeddings, n_clusters=5, text_col="processed_review"):
    """
    df: dataframe aligned row-for-row with `embeddings`
    embeddings: numpy array of shape (len(df), embedding_dim)
    text_col: which column to use for TF-IDF cluster labeling
              (the lemmatized column works BETTER here than for
              semantic embeddings, since TF-IDF is a bag-of-words
              method and benefits from stopword removal)
    """
    n_clusters = max(2, min(n_clusters, len(df) // 5, 10))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    df = df.copy()
    df["cluster"] = cluster_ids

    cluster_names = {}
    for cluster_id in range(n_clusters):
        cluster_texts = df.loc[df["cluster"] == cluster_id, text_col].fillna("")

        if len(cluster_texts) == 0 or cluster_texts.str.len().sum() == 0:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue

        try:
            vectorizer = TfidfVectorizer(max_features=5, stop_words="english")
            vectorizer.fit_transform(cluster_texts)
            top_terms = vectorizer.get_feature_names_out()
            cluster_names[cluster_id] = ", ".join(top_terms) if len(top_terms) > 0 else f"Cluster {cluster_id}"
        except ValueError:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    df["cluster_label"] = df["cluster"].map(cluster_names)

    agg_dict = {"Reviews": ("cluster", "count")}
    if "rating" in df.columns:
        agg_dict["Avg_Rating"] = ("rating", "mean")

    cluster_summary = (
        df.groupby("cluster_label")
        .agg(**agg_dict)
        .round(2)
        .sort_values("Reviews", ascending=False)
    )

    return df, cluster_summary