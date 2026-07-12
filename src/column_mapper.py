"""
ReviewLens AI
Automatic Column Mapper

Two-stage detection:
1. Exact match against a large list of known naming variants per field.
2. Fuzzy fallback (difflib) for anything still unmatched — catches
   naming conventions we didn't anticipate, so a genuinely new CSV
   schema still has a real shot at auto-detecting correctly instead
   of forcing manual mapping every single time.
"""

import difflib

COLUMN_MAPPING = {

    "review_text": [
        "review text", "review", "reviewtext", "text", "content",
        "comment", "feedback", "review_body", "reviewbody", "body",
        "description", "reviews", "customer review", "review comment",
        "opinion", "user_review", "message", "review_message",
        "review_content", "customer_feedback", "comments"
    ],

    "review_title": [
        "title", "headline", "review title", "summary", "review_heading",
        "review_summary", "subject", "review_headline"
    ],

    "rating": [
        "rating", "stars", "star", "score", "overall", "overall rating",
        "ratings", "star_rating", "customer_rating", "review_rating",
        "product_rating", "user_rating"
    ],

    "product_id": [
        "product id", "product_id", "productid", "asin",
        "clothing id", "item id", "product", "sku", "item_id",
        "product_code", "item_sku", "product_sku"
    ],

    "category": [
        "category", "class name", "department name", "division name",
        "product category", "department", "class", "dept", "segment",
        "product_type", "item_category", "product_category_tree"
    ],

    "age": [
        "age", "customer_age", "reviewer_age", "buyer_age"
    ],

    "recommended": [
        "recommended", "recommended ind", "is recommended",
        "recommend", "recommendation", "would_recommend",
        "recommend_product", "positive_recommendation", "does_recommend"
    ],

    "helpful_votes": [
        "helpful votes", "helpful_vote", "helpful_votes",
        "positive feedback count", "likes", "votes", "helpful",
        "upvotes", "helpful_count", "num_helpful", "vote_count"
    ],

    "date": [
        "date", "review date", "reviewdate", "review_date",
        "time", "timestamp", "date posted", "created at",
        "posted_on", "review_timestamp", "created_on",
        "purchase_date", "review_time", "submitted_date"
    ]
}


def normalize(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def detect_columns(df, fuzzy_cutoff=0.85):
    """
    Returns a dict {standard_name: matched_original_column_or_None}.

    fuzzy_cutoff: similarity threshold (0-1) for the fallback fuzzy
    match. Higher = stricter (fewer false positives, more misses).
    0.75 is a reasonable middle ground for short column-name strings.
    """
    detected = {}

    normalized_columns = {
        normalize(col): col
        for col in df.columns
    }
    normalized_keys = list(normalized_columns.keys())
    already_used = set()

    # ---- Stage 1: exact match against known variants ----
    for standard_name, possible_names in COLUMN_MAPPING.items():
        detected[standard_name] = None

        for possible in possible_names:
            key = normalize(possible)

            if key in normalized_columns and normalized_columns[key] not in already_used:
                detected[standard_name] = normalized_columns[key]
                already_used.add(normalized_columns[key])
                break

    # ---- Stage 2: fuzzy fallback for anything still unmatched ----
    for standard_name, possible_names in COLUMN_MAPPING.items():
        if detected[standard_name] is not None:
            continue

        candidate_keys = [k for k in normalized_keys if normalized_columns[k] not in already_used]
        if not candidate_keys:
            continue

        best_match = None
        best_score = 0.0

        for possible in possible_names:
            target = normalize(possible)
            matches = difflib.get_close_matches(target, candidate_keys, n=1, cutoff=fuzzy_cutoff)
            if matches:
                score = difflib.SequenceMatcher(None, target, matches[0]).ratio()
                if score > best_score:
                    best_score = score
                    best_match = matches[0]

        if best_match:
            detected[standard_name] = normalized_columns[best_match]
            already_used.add(normalized_columns[best_match])

    # ---- Stage 3: substring containment, ONLY for the two REQUIRED
    # fields (review_text, rating). These are the fields that force a
    # manual dropdown if undetected, so it's worth being a bit more
    # aggressive here. Optional fields stay strict — a wrong silent
    # match on an optional field is more likely to go unnoticed and
    # quietly produce bad groupings than a required field would. ----
    core_field_hints = {
        "review_text": ["review", "text", "comment", "feedback", "content", "description", "opinion"],
        "rating": ["rating", "star", "score"]
    }

    for standard_name, hints in core_field_hints.items():
        if detected[standard_name] is not None:
            continue

        for key in normalized_keys:
            col = normalized_columns[key]
            if col in already_used:
                continue
            if any(hint in key for hint in hints):
                detected[standard_name] = col
                already_used.add(col)
                break

    return detected