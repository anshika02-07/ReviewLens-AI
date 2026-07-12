"""
ReviewLens AI
RAG (Retrieval-Augmented Generation) pipeline.

Uses Google's Gemini API (free tier, no credit card required).
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_PROMPT = """You are ReviewLens AI's review analysis assistant.

Answer the user's question using ONLY the customer reviews provided as context below.
Rules:
- If the reviews don't contain enough information to answer confidently, say so explicitly rather than guessing.
- When possible, quantify your answer (e.g. "4 of the 8 retrieved reviews mention sizing issues").
- Keep answers concise: 3-5 sentences unless the question asks for a list.
- Do not invent details, product names, or statistics not present in the provided reviews.
"""

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_PROMPT
)


def build_context(retrieved_df, max_reviews=8):
    context_parts = []
    for i, (_, row) in enumerate(retrieved_df.head(max_reviews).iterrows(), start=1):
        rating = row.get("rating", "N/A")
        text = str(row.get("review_text", ""))[:500]
        context_parts.append(f"Review {i} (Rating: {rating}/5): {text}")
    return "\n\n".join(context_parts)


def generate_answer(question, retrieved_df, chat_history=None):
    if retrieved_df is None or len(retrieved_df) == 0:
        return "I couldn't find any reviews relevant to that question. Try rephrasing, or ask about a topic more likely to appear in the dataset (e.g. sizing, quality, delivery)."

    context = build_context(retrieved_df)

    user_message = f"""Customer Reviews Context:
{context}

Question: {question}"""

    history = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    response = chat.send_message(user_message)

    return response.text