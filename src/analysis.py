import pandas as pd
from collections import Counter


def get_common_words(series, top_n=30):

    text = " ".join(series.astype(str))

    words = text.split()

    counter = Counter(words)

    return pd.DataFrame(

        counter.most_common(top_n),

        columns=["Word", "Frequency"]

    )


def analyze_reviews(df):

    """
    Returns all analytics required by Streamlit.
    """

    print("=" * 60)
    print("Generating Analytics...")
    print("=" * 60)

    # =====================================================
    # Overall Summary
    # =====================================================

    summary = {

        "Total Reviews": len(df),

        "Average Rating": round(df["rating"].mean(),2),

        "Average Helpful Votes": round(df["helpful_votes"].mean(),2),

        "Recommendation Rate": round(

            df["recommended"].mean()*100,

            2

        )

    }

    # =====================================================
    # Sentiment Summary
    # =====================================================

    if "sentiment" in df.columns:

        positive = (

            df["sentiment"]=="POSITIVE"

        ).sum()

        negative = (

            df["sentiment"]=="NEGATIVE"

        ).sum()

        summary["Positive Reviews"] = positive

        summary["Negative Reviews"] = negative

        summary["Positive %"] = round(

            positive/len(df)*100,

            2

        )

        summary["Negative %"] = round(

            negative/len(df)*100,

            2

        )

    # =====================================================
    # Rating Distribution
    # =====================================================

    rating_distribution = (

        df["rating"]

        .value_counts()

        .sort_index()

    )

    # =====================================================
    # Category Analysis
    # =====================================================

    category_summary = (

        df.groupby("category")

        .agg(

            Reviews=("rating","count"),

            Average_Rating=("rating","mean"),

            Recommendation=("recommended","mean")

        )

        .round(2)

    )

    category_summary["Recommendation"] *= 100

    # =====================================================
    # Age Analysis
    # =====================================================

    age_summary = None

    if "age" in df.columns:

        bins = [0,25,35,45,55,65,100]

        labels = [

            "18-25",

            "26-35",

            "36-45",

            "46-55",

            "56-65",

            "65+"

        ]

        temp = df.copy()

        temp["Age_Group"] = pd.cut(

            temp["age"],

            bins=bins,

            labels=labels

        )

        age_summary = (

            temp.groupby("Age_Group")

            .agg(

                Reviews=("rating","count"),

                Average_Rating=("rating","mean")

            )

            .round(2)

        )

    # =====================================================
    # Helpful Reviews
    # =====================================================

    top_reviews = (

        df.sort_values(

            "helpful_votes",

            ascending=False

        )

        .head(20)

    )

    # =====================================================
    # Common Words
    # =====================================================

    positive_words = None

    negative_words = None

    if "sentiment" in df.columns:

        positive_words = get_common_words(

            df[

                df["sentiment"]=="POSITIVE"

            ]["processed_review"]

        )

        negative_words = get_common_words(

            df[

                df["sentiment"]=="NEGATIVE"

            ]["processed_review"]

        )

    # =====================================================

    return {

        "summary":summary,

        "rating_distribution":rating_distribution,

        "category_summary":category_summary,

        "age_summary":age_summary,

        "top_reviews":top_reviews,

        "positive_words":positive_words,

        "negative_words":negative_words

    }