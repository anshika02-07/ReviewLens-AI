"""
ReviewLens AI
Executive Summary Generator
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-3.1-flash-lite")


def generate_executive_summary(
    summary_stats,
    top_positive_reviews,
    top_negative_reviews,
    cluster_summary=None,
    trend_description=None
):
    pos_text = "\n".join(f"- {t[:200]}" for t in top_positive_reviews) or "None available"
    neg_text = "\n".join(f"- {t[:200]}" for t in top_negative_reviews) or "None available"

    cluster_text = ""
    if cluster_summary is not None and len(cluster_summary) > 0:
        cluster_text = (
            "\n\nComplaint/Theme Clusters (from K-Means clustering on review embeddings):\n"
            + cluster_summary.to_string()
        )

    if trend_description:
        trend_text = f"\n\nRating trend over time:\n{trend_description}"
    else:
        trend_text = (
            "\n\nNote: This dataset has no usable date/timestamp column, so no "
            "time-based sales or rating trend can be computed. Do NOT speculate "
            "about trends over time — base your analysis only on the content "
            "patterns in the reviews below."
        )

    prompt = f"""You are a business analyst preparing a concise executive summary for a product manager.

Dataset statistics: {summary_stats}

Sample of top positive reviews:
{pos_text}

Sample of top negative reviews:
{neg_text}
{cluster_text}
{trend_text}

Write a structured summary with these exact sections:

**Overall Sentiment Verdict** — 2-3 sentences.

**What Customers Love** — top 3 bullet points, grounded in the reviews shown.

**What Customers Complain About** — top 3 bullet points, grounded in the reviews/clusters shown.

**Why Ratings May Be Moving** — a reasoned hypothesis based ONLY on the patterns in the data above. If there's no time-trend data, say so explicitly and instead explain what content patterns most likely correlate with lower or higher ratings.

Do not invent external causes (marketing campaigns, competitor actions, seasonality) that aren't evidenced in the data provided.
"""

    response = model.generate_content(prompt)
    return response.text