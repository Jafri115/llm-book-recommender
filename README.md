# Semantic Book Recommender System

A content-based book recommendation system using modern NLP techniques and large language models. The system analyzes semantic similarity between book descriptions to help users discover new books based on thematic content and emotional tone.

🚀 **[Live Demo](https://huggingface.co/spaces/Wasifjafri/semantic-book-recommender)** | 📊 **[Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)**

## Features

- **Semantic Search**: Uses OpenAI embeddings and ChromaDB for vector similarity matching
- **Smart Classification**: Zero-shot categorization (fiction/non-fiction) and sentiment analysis
- **Interactive Dashboard**: Gradio interface with book covers and filtering options
- **Rich Dataset**: 6,810 books from Kaggle with descriptions, categories, and ratings

## How It Works

1. **Data Processing**: Cleans book descriptions and handles missing data using LLMs
2. **Vector Embeddings**: Transforms text into semantic vectors using OpenAI API
3. **Similarity Search**: Uses cosine similarity to find thematically similar books
4. **Classification**: Applies zero-shot classification for categories and fine-tuned sentiment analysis

## Installation

### Requirements
- Python 3.8+
- OpenAI API key

### Setup
```bash
git clone https://github.com/yourusername/book-recommender.git
cd book-recommender
pip install -r requirements.txt
```

Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. **Process Data**: Run notebooks in order:
   - `1_data_cleaning.ipynb`
   - `2_vector_database.ipynb`
   - `3_classification.ipynb`
   - `4_sentiment_analysis.ipynb`

2. **Launch App**:
   ```bash
   python app.py
   ```

3. **Access Interface**: Visit `http://localhost:7860`
   - Enter book themes or descriptions
   - Apply category and emotional tone filters
   - Browse personalized recommendations

## Tech Stack

- **Python** (Pandas, NumPy)
- **OpenAI API** (Embeddings)
- **ChromaDB** (Vector database)
- **Hugging Face** (NLP models)
- **LangChain** (LLM integration)
- **Gradio** (Web interface)

## Key Learnings

- Practical experience with LLM applications and vector databases
- End-to-end ML pipeline from data cleaning to deployment
- Understanding of semantic search and text classification techniques

## Future Improvements

- Add collaborative filtering for hybrid recommendations
- Expand to include more book genres and categories
- Deploy as web service with user accounts

## Acknowledgments

Developed following tutorial by Jodie Burchell (JetBrains). MIT License.

---
title: {{title}}
emoji: {{emoji}}
colorFrom: {{colorFrom}}
colorTo: {{colorTo}}
sdk: {{sdk}}
sdk_version: "{{sdkVersion}}"
app_file: app.py
pinned: false
---

