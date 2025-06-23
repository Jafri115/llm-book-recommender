import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr

# Load environment variables (if any)
load_dotenv()

# --- 1. Load your book metadata ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].str.contains("nan"),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --- 2. Build embeddings DB ---
raw_docs = TextLoader("tagged_description.txt", encoding="utf-8").load()
splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = splitter.split_documents(raw_docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
)
db_books = Chroma.from_documents(documents, embedding_model)

# --- 3. Recommendation logic ---
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    isbns = [int(r.page_content.strip('"').split()[0]) for r in recs]
    df = books[books["isbn13"].isin(isbns)].head(initial_top_k)

    if category and category != "All":
        df = df[df["simple_categories"] == category].head(final_top_k)
    else:
        df = df.head(final_top_k)

    tone_map = {
        "Happy":      "joy",
        "Surprising": "surprise",
        "Angry":      "anger",
        "Suspenseful":"fear",
        "Sad":        "sadness",
    }
    if tone in tone_map:
        df = df.sort_values(by=tone_map[tone], ascending=False).head(final_top_k)
    return df

def recommend_books(query: str, category: str, tone: str):
    df = retrieve_semantic_recommendations(query, category, tone)
    gallery_results = []
    for _, row in df.iterrows():
        img = row["large_thumbnail"] or "cover-not-found.jpg"
        title = row["title"] or "Unknown Title"
        authors = row["authors"] or "Unknown Author"
        desc = row["description"] or ""
        # Format authors
        if ";" in authors:
            parts = [a.strip() for a in authors.split(";")]
            if len(parts) == 2:
                authors_str = f"{parts[0]} and {parts[1]}"
            else:
                authors_str = ", ".join(parts[:-1]) + ", and " + parts[-1]
        else:
            authors_str = authors
        # Truncate description
        truncated = " ".join(desc.split()[:30]) + "..."
        caption = f"{title} by {authors_str}: {truncated}"
        # **Return as [image, caption] lists**
        gallery_results.append([img, caption])
    return gallery_results

# --- 4. Gradio UI ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones      = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")
    with gr.Row():
        q_in = gr.Textbox(label="Describe a book you'd like")
        cat = gr.Dropdown(choices=categories, value="All", label="Category")
        tone = gr.Dropdown(choices=tones, value="All", label="Tone")
    gallery = gr.Gallery(
        label="Recommendations",
        columns=4,
        object_fit="contain",
        preview=True
    )
    btn = gr.Button("Search")
    btn.click(
        fn=recommend_books,
        inputs=[q_in, cat, tone],
        outputs=[gallery]
    )

# --- 5. Launch ---
if __name__ == "__main__":
    dashboard.launch()