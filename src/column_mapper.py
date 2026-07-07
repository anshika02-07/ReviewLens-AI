"""
ReviewLens AI
Automatic Column Mapper
"""

COLUMN_MAPPING = {

    "review_text": [
        "review text",
        "review",
        "reviewtext",
        "text",
        "content",
        "comment",
        "feedback",
        "review_body",
        "body",
        "description"
    ],

    "review_title": [
        "title",
        "headline",
        "review title"
    ],

    "rating": [
        "rating",
        "stars",
        "star",
        "score",
        "overall",
        "overall rating"
    ],

    "product_id": [
        "product id",
        "product_id",
        "productid",
        "asin",
        "clothing id",
        "item id",
        "product"
    ],

    "category": [
        "category",
        "class name",
        "department name",
        "division name",
        "product category",
        "department",
        "class"
    ],

    "age": [
        "age"
    ],

    "recommended": [
        "recommended",
        "recommended ind",
        "is recommended",
        "recommend",
        "recommendation"
    ],

    "helpful_votes": [
        "helpful votes",
        "helpful_vote",
        "helpful_votes",
        "positive feedback count",
        "likes",
        "votes",
        "helpful"
    ]
}


def normalize(name):
    """
    Makes column matching robust.

    Example:
    Review Text
    review_text
    review-text
    ReviewText

    All become

    reviewtext
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def detect_columns(df):

    detected = {}

    normalized_columns = {
        normalize(col): col
        for col in df.columns
    }

    for standard_name, possible_names in COLUMN_MAPPING.items():

        detected[standard_name] = None

        for possible in possible_names:

            key = normalize(possible)

            if key in normalized_columns:

                detected[standard_name] = normalized_columns[key]
                break

    return detected