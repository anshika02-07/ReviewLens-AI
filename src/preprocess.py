import re
import spacy

print("Loading spaCy model...")

nlp = spacy.load("en_core_web_sm")


def preprocess(text):
    """
    Clean a single review.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    doc = nlp(text)

    words = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        lemma = token.lemma_.strip()

        if lemma == "":
            continue

        words.append(lemma)

    return " ".join(words)


def preprocess_reviews(df):
    """
    Preprocess all reviews inside dataframe.
    """

    print("=" * 60)
    print("Preprocessing Reviews...")
    print("=" * 60)

    if "review_text" not in df.columns:
        raise ValueError("Column 'review_text' not found.")

    # Don't preprocess twice
    if "processed_review" not in df.columns:

        df["processed_review"] = df["review_text"].apply(preprocess)

    df = df[df["processed_review"].str.len() > 2]

    df.reset_index(drop=True, inplace=True)

    print(f"Processed Reviews : {len(df)}")

    return df