import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from gradio.themes.base import Base
import logging
# from __future__ import annotations
from typing import Iterable
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables (if any)
load_dotenv()

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_lg,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Enhanced background with subtle gradient
            body_background_fill="linear-gradient(135deg, #f0fdfa 0%, #3a3638 50%, #3a3638 100%)",
            body_background_fill_dark="linear-gradient(135deg, #064e3b 0%, #0c4a6e 50%, #065f46 100%)",
            
            # Enhanced button styling - reduced padding
            button_primary_background_fill="linear-gradient(135deg, *primary_500 0%, *secondary_600 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, *primary_600 0%, *secondary_700 100%)",
            button_primary_background_fill_dark="linear-gradient(135deg, *primary_600 0%, *secondary_700 100%)",
            button_primary_text_color="white",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="12px 24px",  # Reduced from 20px 40px
            
            # Block styling
            block_background_fill="rgba(30, 41, 59, 0.95)",
            block_background_fill_dark="rgba(30, 41, 59, 0.95)",
            block_border_width="1px",
            block_border_color="rgba(16, 185, 129, 0.3)",
            block_shadow="*shadow_drop_lg",
            block_title_text_weight="700",
            block_title_text_color="#34d399",
            block_title_text_color_dark="#34d399",
            
            # Input styling  
            input_background_fill="rgba(30, 41, 59, 0.9)",
            input_background_fill_dark="rgba(30, 41, 59, 0.9)",
            input_border_width="1px",
            input_shadow="*shadow_drop",
            input_shadow_focus="*shadow_drop_lg",
            
            # Slider improvements
            slider_color="*primary_500",
            slider_color_dark="*primary_400",
            
            # Panel styling
            panel_background_fill="rgba(30, 41, 59, 0.95)",
            panel_background_fill_dark="rgba(30, 41, 59, 0.95)",
        )

seafoam = Seafoam()

# --- 1. Load your book metadata ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].str.contains("nan"),
    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDEyMCAxODAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTgwIiBmaWxsPSIjMzMzIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMiIvPgo8dGV4dCB4PSI2MCIgeT0iOTAiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIENvdmVyPC90ZXh0Pgo8L3N2Zz4=",
    books["large_thumbnail"]
)

# --- 2. Build embeddings DB ---
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},  # Use CPU for Hugging Face Spaces
)
PERSIST_DIR = "chroma_db"

if os.path.exists(PERSIST_DIR):
    db_books = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
else:
    raw_docs = TextLoader("tagged_description.txt", encoding="utf-8").load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=0)
    documents = splitter.split_documents(raw_docs)
    db_books = Chroma.from_documents(documents, embedding_model, persist_directory=PERSIST_DIR)
    db_books.persist()

# --- Recommender ---
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    isbns = [int(r.page_content.strip('"').split()[0]) for r in recs]
    df = books[books["isbn13"].isin(isbns)].head(initial_top_k)

    if category and category != "All":
        df = df[df["simple_categories"] == category].head(final_top_k)

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

def recommend_books(query, category, tone):
    df = retrieve_semantic_recommendations(query, category, tone)
    gallery_results = []
    for _, row in df.iterrows():
        img = row["large_thumbnail"] or "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDEyMCAxODAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTgwIiBmaWxsPSIjMzMzIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMiIvPgo8dGV4dCB4PSI2MCIgeT0iOTAiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIENvdmVyPC90ZXh0Pgo8L3N2Zz4="
        title = row["title"] or "Unknown Title"
        authors = row["authors"] or "Unknown Author"
        desc = row["description"] or ""
        # Format authors
        if ";" in authors:
            parts = [a.strip() for a in authors.split(";")]
            authors_str = f"{parts[0]} and {parts[1]}" if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]
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

with gr.Blocks(theme=seafoam, title="Book Recommender", css="""
    /* Additional custom CSS for enhanced styling */
    .gradio-container {
        background: linear-gradient(135deg, #3a3638 0%, #3a3638 50%, #3a3638 100%) !important;
        min-height: 100vh;
    }
    
    /* Enhanced input focus effects */
    .gr-textbox:focus-within,
    .gr-dropdown:focus-within {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Button hover animations and width control */
    .gr-button:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Control button width to fit text on one line */
    .gr-button {
        width: auto !important;
        min-width: fit-content !important;
        max-width: 250px !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        overflow: hidden !important;
    }
    
    /* Gallery card styling */
    .gr-gallery .grid-item {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .gr-gallery .grid-item:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Title styling */
    .gr-markdown h1 {
        background: linear-gradient(135deg, #059669 0%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtle animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .gr-block {
        animation: fadeIn 0.6s ease-out;
    }
""") as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")
    with gr.Row():
        q_in = gr.Textbox(
            label="Describe a book you'd like", 
            placeholder="e.g., 'a thrilling mystery novel' or 'a heartwarming romance'",
            container=True,
            scale=3
        )
        cat = gr.Dropdown(
            choices=categories, 
            value="All", 
            label="Category",
            container=True,
            scale=1
        )
        tone = gr.Dropdown(
            choices=tones, 
            value="All", 
            label="Tone",
            container=True,
            scale=1
        )
    
    # Center the button and reduce its width
    with gr.Row():
        with gr.Column(scale=1):
            pass  # Empty column for spacing
        with gr.Column(scale=1):
            btn = gr.Button(
                "🔍 Search for Books", 
                variant="primary", 
                size="lg"
            )
        with gr.Column(scale=1):
            pass  # Empty column for spacing
    
    gallery = gr.Gallery(
        label="📖 Your Personalized Recommendations",
        columns=4,
        object_fit="contain",
        preview=True,
        container=True,
        height="auto"
    )
    
    btn.click(
        fn=recommend_books,
        inputs=[q_in, cat, tone],
        outputs=[gallery]
    )

if __name__ == "__main__":
    dashboard.launch()
