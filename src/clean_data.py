import pandas as pd


def clean_dataset(df):

    """
    Cleans a standardized dataframe.

    Expected columns:
    review_text
    review_title
    rating
    recommended
    helpful_votes
    category
    age
    product_id
    """

    print("=" * 60)
    print("Cleaning Dataset...")
    print("=" * 60)

    print(f"Original Shape : {df.shape}")

    # -------------------------------
    # Remove rows without review text
    # -------------------------------

    df = df.dropna(subset=["review_text"])

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
    # Reset index
    # -------------------------------

    df.reset_index(drop=True, inplace=True)

    print(f"Final Shape : {df.shape}")

    return df