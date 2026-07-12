import pandas as pd
from transformers import pipeline

# =====================================================
# Load Model (Loads Only Once)
# =====================================================

print("=" * 60)
print("Loading DistilBERT Sentiment Model...")
print("=" * 60)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def predict_sentiment(text):
    """
    Predict sentiment for a single review.
    """
    try:
        result = classifier(str(text)[:512])[0]
        return result["label"], round(result["score"], 4)
    except Exception:
        return "UNKNOWN", 0.0


def run_sentiment(df, progress_callback=None):
    """
    Runs DistilBERT on dataframe.

    progress_callback: optional function(fraction: float) called
    periodically so the caller (Streamlit) can show a live progress
    bar instead of a spinner with no feedback. Without this, large
    samples can look "stuck" for minutes even though they're working.
    """
    print("=" * 60)
    print("Running DistilBERT Sentiment Analysis...")
    print("=" * 60)

    from src.preprocess import preprocess_reviews

    if "processed_review" not in df.columns:
        df = preprocess_reviews(df)

    if "sentiment" in df.columns and "confidence" in df.columns:
        print("Sentiment already exists.")
        return df

    total = len(df)
    labels = []
    scores = []

    for i, text in enumerate(df["processed_review"]):
        label, score = predict_sentiment(text)
        labels.append(label)
        scores.append(score)

        # Update every 10 rows (or on the last row) — updating on
        # every single row can itself slow Streamlit down.
        if progress_callback and (i % 10 == 0 or i == total - 1):
            progress_callback((i + 1) / total)

    df["sentiment"] = labels
    df["confidence"] = scores

    print("\nSentiment Distribution\n")
    print(df["sentiment"].value_counts())

    return df