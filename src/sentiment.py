import pandas as pd
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas()

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

# =====================================================
# Predict One Review
# =====================================================

def predict_sentiment(text):
    """
    Predict sentiment for a single review.
    """

    try:

        result = classifier(str(text)[:512])[0]

        return pd.Series([
            result["label"],
            round(result["score"], 4)
        ])

    except Exception:

        return pd.Series([
            "UNKNOWN",
            0.0
        ])

# =====================================================
# Run Sentiment Analysis
# =====================================================

def run_sentiment(df):
    """
    Runs DistilBERT on dataframe.
    """

    print("=" * 60)
    print("Running DistilBERT Sentiment Analysis...")
    print("=" * 60)

    from src.preprocess import preprocess_reviews

    if "processed_review" not in df.columns:

        df = preprocess_reviews(df)

    # Don't run twice
    if "sentiment" in df.columns and "confidence" in df.columns:

        print("Sentiment already exists.")

        return df

    df[["sentiment", "confidence"]] = (

        df["processed_review"]

        .progress_apply(predict_sentiment)

    )

    print("\nSentiment Distribution\n")

    print(df["sentiment"].value_counts())

    return df