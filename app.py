import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import time
from gradio.themes.base import Base
import logging
from typing import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class LibraryTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.amber,
        secondary_hue: colors.Color | str = colors.orange,
        neutral_hue: colors.Color | str = colors.stone,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Crimson Text"),
            "serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("JetBrains Mono"),
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
            # Dark library background
            body_background_fill="#2a1810",
            body_background_fill_dark="#2a1810",
            
            # Button styling
            button_primary_background_fill="linear-gradient(135deg, #d97706 0%, #ea580c 100%)",
            button_primary_background_fill_hover="linear-gradient(135deg, #ea580c 0%, #dc2626 100%)",
            button_primary_text_color="white",
            button_primary_border_color="transparent",
            button_large_padding="16px 32px",
            button_large_text_size="18px",
            
            # Block styling for cards
            block_background_fill="rgba(68, 54, 40, 0.9)",
            block_background_fill_dark="rgba(68, 54, 40, 0.9)",
            block_border_width="1px",
            block_border_color="rgba(217, 119, 6, 0.3)",
            block_radius="12px",
            
            # Input styling
            input_background_fill="rgba(41, 37, 36, 0.8)",
            input_background_fill_dark="rgba(41, 37, 36, 0.8)",
            input_border_color="rgba(217, 119, 6, 0.4)",
            input_text_color="#f5f5f4",
            input_placeholder_color="rgba(245, 245, 244, 0.6)",
            
            # Text colors
            body_text_color="#f5f5f4",
            body_text_color_dark="#f5f5f4",
            
            # Panel styling
            panel_background_fill="rgba(68, 54, 40, 0.9)",
            panel_background_fill_dark="rgba(68, 54, 40, 0.9)",
        )

library_theme = LibraryTheme()

# --- 1. Load your book metadata ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].str.contains("nan"),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --- 2. Build embeddings DB ---
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
)
PERSIST_DIR = "chroma_db"
if os.path.exists(PERSIST_DIR):
    logger.info(f"Loading existing Chroma DB from {PERSIST_DIR}...")
    db_books = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )
else:
    logger.info(f"Creating new Chroma DB in {PERSIST_DIR}...")
    raw_docs = TextLoader("tagged_description.txt", encoding="utf-8").load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=0)
    documents = splitter.split_documents(raw_docs)
    db_books = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=PERSIST_DIR
    )
    db_books.persist()

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
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_map:
        df = df.sort_values(by=tone_map[tone], ascending=False).head(final_top_k)
    return df

def recommend_books(query: str, category: str, tone: str):
    df = retrieve_semantic_recommendations(query, category, tone)
    
    # Create HTML cards instead of gallery
    cards_html = """
    <div class="book-grid">
    """
    
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
        truncated_desc = " ".join(desc.split()[:25]) + "..."
        
        card_html = f"""
        <div class="book-card">
            <div class="book-cover">
                <img src="{img}" alt="{title}" onerror="this.src='cover-not-found.jpg'">
            </div>
            <div class="book-info">
                <h3 class="book-title">{title}</h3>
                <p class="book-author">{authors_str}</p>
                <p class="book-description">{truncated_desc}</p>
            </div>
        </div>
        """
        cards_html += card_html
    
    cards_html += "</div>"
    return cards_html

# --- 4. Gradio UI ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS for library theme
library_css = """
/* Library background with warm wood tones */
.gradio-container {
    background: linear-gradient(135deg, #2a1810 0%, #3d2914 50%, #2a1810 100%) !important;
    background-image: 
        radial-gradient(circle at 20% 20%, rgba(217, 119, 6, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(234, 88, 12, 0.1) 0%, transparent 50%);
    min-height: 100vh;
    font-family: 'Crimson Text', serif;
}

/* Header styling */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: rgba(68, 54, 40, 0.3);
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(217, 119, 6, 0.2);
}

.main-header h1 {
    color: #fbbf24;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}

.book-icon {
    font-size: 2.5rem;
    color: #d97706;
}

/* Search section styling */
.search-section {
    background: rgba(68, 54, 40, 0.6);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid rgba(217, 119, 6, 0.3);
    margin-bottom: 2rem;
}

.search-row {
    display: flex;
    gap: 1rem;
    align-items: end;
    margin-bottom: 1.5rem;
}

.search-input {
    flex: 2;
}

.search-dropdown {
    flex: 1;
}

/* Input styling */
.gr-textbox input,
.gr-dropdown .wrap {
    background: rgba(41, 37, 36, 0.8) !important;
    border: 1px solid rgba(217, 119, 6, 0.4) !important;
    color: #f5f5f4 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
}

.gr-textbox input::placeholder {
    color: rgba(245, 245, 244, 0.6) !important;
    font-style: italic;
}

/* Button styling */
.search-button {
    background: linear-gradient(135deg, #d97706 0%, #ea580c 100%) !important;
    border: none !important;
    color: white !important;
    padding: 16px 32px !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3) !important;
}

.search-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(217, 119, 6, 0.4) !important;
    background: linear-gradient(135deg, #ea580c 0%, #dc2626 100%) !important;
}

/* Book grid styling */
.book-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 2rem;
    padding: 2rem 0;
}

.book-card {
    background: rgba(68, 54, 40, 0.9);
    border: 1px solid rgba(217, 119, 6, 0.3);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.book-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
    border-color: rgba(217, 119, 6, 0.6);
}

.book-cover {
    width: 100%;
    height: 320px;
    overflow: hidden;
    position: relative;
}

.book-cover img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.3s ease;
}

.book-card:hover .book-cover img {
    transform: scale(1.05);
}

.book-info {
    padding: 1.5rem;
}

.book-title {
    color: #fbbf24;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    line-height: 1.3;
    font-style: italic;
}

.book-author {
    color: #d97706;
    font-size: 1rem;
    font-weight: 500;
    margin: 0 0 1rem 0;
}

.book-description {
    color: #e7e5e4;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0;
    opacity: 0.9;
}

/* Results section styling */
.results-section {
    background: rgba(68, 54, 40, 0.3);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid rgba(217, 119, 6, 0.2);
}

.results-header {
    color: #fbbf24;
    font-size: 1.75rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    text-align: center;
    font-style: italic;
}

/* Label styling */
.gr-box label {
    color: #fbbf24 !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.book-card {
    animation: fadeIn 0.6s ease-out;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(41, 37, 36, 0.3);
}

::-webkit-scrollbar-thumb {
    background: rgba(217, 119, 6, 0.6);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(217, 119, 6, 0.8);
}


with gr.Blocks(theme=library_theme, title="Book Recommendation System", css=library_css) as dashboard:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1><span class="book-icon">📚</span> Book Recommendation System</h1>
    </div>
    """)
    
    # Search Section
    with gr.Column(elem_classes="search-section"):
        with gr.Row(elem_classes="search-row"):
            query_input = gr.Textbox(
                label="Describe a book you'd like",
                placeholder="e.g., 'a mystery with unreliable narrator' or 'fantasy with strong female lead'",
                elem_classes="search-input"
            )
            category_dropdown = gr.Dropdown(
                choices=categories,
                value="All",
                label="Genre",
                elem_classes="search-dropdown"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                value="All",
                label="Reading Mood",
                elem_classes="search-dropdown"
            )
        
        with gr.Row():
            search_btn = gr.Button(
                "🔍 Find Your Next Read",
                variant="primary",
                size="lg",
                elem_classes="search-button"
            )
    
    # Results Section
    with gr.Column(elem_classes="results-section"):
        gr.HTML('<div class="results-header">Your Bookshelf Recommendations</div>')
        results_display = gr.HTML()
    
    # Connect the search function
    search_btn.click(
        fn=recommend_books,
        inputs=[query_input, category_dropdown, tone_dropdown],
        outputs=[results_display]
    )

# --- 5. Launch ---
if __name__ == "__main__":
    dashboard.launch()
