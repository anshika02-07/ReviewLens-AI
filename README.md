# ReviewLens AI

AI-powered customer review analytics platform built using Python and Streamlit.

## Demo

![Dashboard](assets/dashboard.png.png)

## Features

- Upload any review dataset (CSV)
- Automatic column detection
- Manual column mapping
- Interactive dashboard
- Rating distribution
- Category-wise analysis
- Customer age analysis
- Search reviews
- Download processed dataset

## Screenshots

**Rating Distribution & Dataset Overview**
![Charts](assets/charts.png.png)

**Keyword Search**
![Keyword Search](assets/keyword_search.png.png)

**Recommendation Breakdown**
![Recommendation](assets/recommendation.png.png)

## Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- spaCy
- Transformers
- Sentence Transformers
- FAISS

## Setup & Run

```bash
git clone https://github.com/anshika02-07/reviewlens-ai.git
cd reviewlens-ai
pip install -r requirements.txt
streamlit run app.py
```

## Future Enhancements

- Semantic Search
- RAG Chatbot
- LLM-based Review Assistant
