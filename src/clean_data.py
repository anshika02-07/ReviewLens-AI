import pandas as pd


def clean_dataset(df):
    """
    Cleans a standardized dataframe.

    Expected columns:
    review_text, review_title, rating, recommended,
    helpful_votes, category, age, product_id
    """

    print("=" * 60)
    print("Cleaning Dataset...")
    print("=" * 60)
    print(f"Original Shape : {df.shape}")

    # -------------------------------
    # Remove rows without review text
    # -------------------------------
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)

    # -------------------------------
    # Remove duplicate reviews
    # -------------------------------
    df = df.drop_duplicates(subset=["review_text"])

    # -------------------------------
    # Fill optional columns
    # -------------------------------
    optional_columns = {
        "review_title": "",
        "category": "Unknown",
        "helpful_votes": 0,
        "recommended": 0,
        "age": -1,
        "product_id": "Unknown"
    }

    for column, value in optional_columns.items():
        if column not in df.columns:
            df[column] = value
        else:
            df[column] = df[column].fillna(value)

    # -------------------------------
    # Force numeric columns to actually be numeric.
    # Different CSVs may have ratings as "5.0", "5", "Five", etc.
    # Anything that can't be parsed becomes NaN, then dropped/filled,
    # so the app never crashes on a differently-typed dataset.
    # -------------------------------
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(-1)

    # "recommended" may arrive as True/False, 1/0, "Yes"/"No" — normalize to 0/1
    if df["recommended"].dtype == object:
        df["recommended"] = (
            df["recommended"]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
            .fillna(0)
        )
    df["recommended"] = pd.to_numeric(df["recommended"], errors="coerce").fillna(0)

    # -------------------------------
    # Reset index
    # -------------------------------
    df.reset_index(drop=True, inplace=True)

    print(f"Final Shape : {df.shape}")

    return df