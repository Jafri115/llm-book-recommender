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
import base64

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Data ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].str.contains("nan"),
    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDEyMCAxODAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTgwIiBmaWxsPSIjMzMzIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMiIvPgo8dGV4dCB4PSI2MCIgeT0iOTAiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIENvdmVyPC90ZXh0Pgo8L3N2Zz4=",
    books["large_thumbnail"]
)

# --- Vector DB ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cpu"})
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
        "Light & Fun": "joy",
        "Surprising": "surprise",
        "Dark & Intense": "anger",
        "Suspenseful": "fear",
        "Emotional": "sadness",
    }
    if tone in tone_map:
        df = df.sort_values(by=tone_map[tone], ascending=False).head(final_top_k)
    return df

def recommend_books(query, category, tone):
    df = retrieve_semantic_recommendations(query, category, tone)
    if df.empty:
        return """
        <div class='empty-state'>
            <h3>📚 No books found</h3>
            <p>Try a different description or filter to discover your next great read.</p>
        </div>
        """

    cards_html = "<div class='books-grid'>"
    for _, row in df.iterrows():
        img = row["large_thumbnail"] or "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDEyMCAxODAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iMTgwIiBmaWxsPSIjMzMzIiBzdHJva2U9IiM2NjYiIHN0cm9rZS13aWR0aD0iMiIvPgo8dGV4dCB4PSI2MCIgeT0iOTAiIGZpbGw9IndoaXRlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPk5vIENvdmVyPC90ZXh0Pgo8L3N2Zz4="
        title = row["title"] or "Unknown Title"
        authors = row["authors"] or "Unknown Author"
        desc = row["description"] or ""
        truncated_desc = " ".join(desc.split()[:25]) + "..." if len(desc.split()) > 25 else desc

        # Fix the TypeError: Check if authors is a string and not NaN
        if pd.isna(authors) or not isinstance(authors, str):
            authors_str = "Unknown Author"
        elif ";" in authors:
            parts = [a.strip() for a in authors.split(";")]
            authors_str = f"{parts[0]} and {parts[1]}" if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]
        else:
            authors_str = authors

        cards_html += f"""
        <div class='book-card'>
            <div class='book-cover'>
                <img src='{img}' alt='{title}' loading='lazy'>
            </div>
            <div class='book-title'>{title}</div>
            <div class='book-author'>{authors_str}</div>
            <div class='book-description'>{truncated_desc}</div>
        </div>
        """
    cards_html += "</div>"
    return cards_html

# --- Custom CSS (Fixed background gradient) ---
custom_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

/* Global styling */
.gradio-container {
    font-family: 'Crimson Text', Georgia, serif !important;
    background: url('https://i.postimg.cc/XvDdDzBX/bg.png') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    color: #f4e4c1 !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* Background overlay - REMOVED */

/* Main header */
.main-header {
    text-align: center;
    margin-bottom: 20px;
    position: relative;
    z-index: 10;
}

.main-header h1 {
    font-size: 2.5rem;
    color: #d4af37;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 15px;
}

.main-header h1 i {
    font-size: 2.2rem;
    color: #d4af37;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}


/* Input section styling - REDUCED PADDING */
.input-section {
    background: #0F3D2D !important;
    padding: 5px !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    border: none solid #1D372A;
    position: relative;
    z-index: 10;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Form inputs */
.input-section label {
    color: #f4e4c1 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    margin-bottom: 3px !important;
}

.input-section input,
.input-section select {
    background: rgba(26, 77, 58, 0.8) !important;
    border: 2px solid #1D372A !important;
    border-radius: 6px !important;
    color: #f4e4c1 !important;
    font-size: 0.9rem !important;
    padding: 6px 8px !important;
    transition: all 0.3s ease !important;
    margin-bottom: 2px !important;
}

.input-section input:focus,
.input-section select:focus {
    border-color: #6db584 !important;
    box-shadow: 0 0 10px rgba(109, 181, 132, 0.3) !important;
}

.input-section input::placeholder {
    color: #a0a0a0 !important;
    font-style: italic !important;
}

/* Search button - REDUCED SPACING */
.search-button {
    background: linear-gradient(135deg, #2d6b4a 0%, #3a8f5a 100%) !important;
    color: white !important;
    padding: 8px 16px !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin: 8px auto !important;
    display: block !important;
    min-width: unset !important;
    width: auto !important;
    padding: 8px 12px !important;
}
# .search-button {
#     background: linear-gradient(90deg, #0f6a75, #1663b0); /* Blue gradient */
#     color: white;
#     padding: 10px 18px;
#     border: none;
#     border-radius: 12px;
#     font-size: 1rem;
#     font-weight: 600;
#     cursor: pointer;
#     display: inline-flex;
#     align-items: center;
#     gap: 8px; /* space between icon and text */
#     box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
#     transition: all 0.3s ease;
#     min-width: unset !important;
#     width: auto !important;
#     padding: 8px 12px !important;
# }
.search-button:hover {
    background: linear-gradient(135deg, #3a8f5a 0%, #4db36a 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(58, 143, 90, 0.4) !important;
}

/* Results section */
.results-section {
    position: relative;
    z-index: 10;
    margin-top: 20px;
}

/* Books grid */
.books-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 30px;
    padding: 20px 0;
}

.book-card {
    background: linear-gradient(135deg, #f4e4c1 0%, #e6d7b8 100%) !important;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    border: 3px solid #d4af37;
    animation: fadeInUp 0.6s ease both;
}

.book-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
}

.book-cover {
    width: 180px;
    height: 250px;
    margin: 0 auto 20px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    position: relative;
}

.book-cover img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.book-title {
    font-size: 1.3rem;
    font-weight: bold;
    color: #2c1810;
    margin-bottom: 8px;
    font-style: italic;
}

.book-author {
    font-size: 1.1rem;
    color: #5d4037;
    margin-bottom: 12px;
    font-weight: 500;
}

.book-description {
    font-size: 0.95rem;
    color: #4a3426;
    line-height: 1.4;
    text-align: left;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #f4e4c1 0%, #e6d7b8 100%) !important;
    border-radius: 15px;
    border: 3px solid #d4af37;
    margin: 20px 0;
}

.empty-state h3 {
    color: #2c1810;
    font-size: 1.5rem;
    margin-bottom: 10px;
}

.empty-state p {
    color: #4a3426;
    font-size: 1.1rem;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive design */
@media (max-width: 768px) {
    .books-grid {
        grid-template-columns: 1fr;
    }
    
    .main-header h1 {
        font-size: 2rem;
    }
    
    .input-section {
        padding: 8px !important;
        max-width: 95%;
    }
}

/* Override Gradio default styles */
.gradio-container .wrap {
    background: transparent !important;
}

.gradio-container .container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* Comprehensive Grey Background Overrides for Gradio */

/* Override Gradio's default grey backgrounds */
.gradio-container .block {
    background: transparent !important;
}

.gradio-container .form {
    background: transparent !important;
}

.gradio-container .panel {
    background: transparent !important;
}

/* Target specific grey color values */
[style*="#2B2B2B"], 
[style*="#2b2b2b"], 
[style*="rgb(43, 43, 43)"], 
[style*="rgba(43, 43, 43"],
[style*="#1f2937"],
[style*="#374151"],
[style*="#4b5563"],
[style*="#6b7280"],
[style*="#9ca3af"] {
    background: rgba(26, 77, 58, 0.8) !important;
}

/* Override common Gradio component backgrounds */
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .group,
.gradio-container .block-container,
.gradio-container .form-container {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}

/* Input field backgrounds */
.gradio-container input,
.gradio-container select,
.gradio-container textarea {
    background: #0F3D2D !important;
    border: none solid #1D372A !important;
    color: #f4e4c1 !important;
    padding: 3px 8px !important;
    font-size: 0.9rem !important;
}

/* Button backgrounds */
.gradio-container button {
    background: linear-gradient(135deg, #2d6b4a 0%, #3a8f5a 100%) !important;
    color: white !important;
    border: none !important;
}

.gradio-container button:hover {
    background: linear-gradient(135deg, #3a8f5a 0%, #4db36a 100%) !important;
}

/* Tab backgrounds */
.gradio-container .tab-nav,
.gradio-container .tabitem {
    background: transparent !important;
}

/* Dropdown backgrounds */
.gradio-container .dropdown {
    background: rgba(26, 77, 58, 0.8) !important;
}

/* Dropdown options list - should have solid background for readability */
.gradio-container .dropdown-menu,
.gradio-container .dropdown-content,
.gradio-container select option,
.gradio-container .choices__list,
.gradio-container .choices__item,
.gradio-container [role="listbox"],
.gradio-container [role="option"] {
    background: #1a4d3a !important;
    color: #f4e4c1 !important;
    border: 0px solid #1D372A !important;
}

/* Dropdown option hover states */
.gradio-container .dropdown-menu li:hover,
.gradio-container .dropdown-content li:hover,
.gradio-container select option:hover,
.gradio-container .choices__item:hover,
.gradio-container [role="option"]:hover {
    background: #2d6b4a !important;
    color: #ffffff !important;
}

/* Modal and popup backgrounds */
.gradio-container .modal,
.gradio-container .popup {
    background: rgba(26, 77, 58, 0.95) !important;
}

/* Accordion backgrounds */
.gradio-container .accordion {
    background: transparent !important;
}

/* Progress bar backgrounds */
.gradio-container .progress {
    background: rgba(26, 77, 58, 0.8) !important;
}

/* Radio and checkbox backgrounds */
.gradio-container .radio,
.gradio-container .checkbox {
    background: transparent !important;
}

/* Slider backgrounds */
.gradio-container .slider {
    background: rgba(26, 77, 58, 0.8) !important;
}

/* File upload backgrounds */
.gradio-container .file-upload {
    background: rgba(26, 77, 58, 0.8) !important;
    border: 0px dashed #1D372A !important;
}

/* Error message backgrounds */
.gradio-container .error {
    background: rgba(139, 69, 19, 0.8) !important;
    color: #f4e4c1 !important;
}

/* Success message backgrounds */
.gradio-container .success {
    background: rgba(26, 77, 58, 0.8) !important;
    color: #f4e4c1 !important;
}

/* Additional fallback overrides */
.gradio-container [class*="bg-gray"],
.gradio-container [class*="bg-grey"],
.gradio-container [class*="bg-slate"],
.gradio-container [class*="bg-zinc"] {
    background: rgba(26, 77, 58, 0.8) !important;
}

/* Override any remaining grey backgrounds with important declarations */
.gradio-container * {
    background-color: transparent !important;
}

/* Re-apply your custom backgrounds with higher specificity */
.gradio-container .input-section {
    background: #0F3D2D !important;
}

.gradio-container .book-card {
    background: linear-gradient(135deg, #f4e4c1 0%, #e6d7b8 100%) !important;
}

.gradio-container .empty-state {
    background: linear-gradient(135deg, #f4e4c1 0%, #e6d7b8 100%) !important;
}

/* Ensure main container background is preserved - NO OVERLAYS */
.gradio-container {
    background: url('https://i.postimg.cc/XvDdDzBX/bg.png') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}
"""

# --- UI ---
categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Light & Fun", "Surprising", "Dark & Intense", "Suspenseful", "Emotional"]

with gr.Blocks(css=custom_css, theme=Base(), title="Book Recommendation System") as dashboard:
    gr.HTML("""
        <div class='main-header'>
            <h1><i class="fas fa-book-open"></i> Book Recommendation System</h1>
        </div>
    """)

    with gr.Column(elem_classes=["input-section"]):
        description_input = gr.Textbox(
            label="Describe a book you'd like",
            placeholder="e.g., 'a mystery with unreliable narrator' or 'fantasy with strong female lead'",
            lines=2,
            elem_classes=["description-input"]
        )
        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=categories,
                value="All",
                label="Genre",
                elem_classes=["genre-dropdown"]
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                value="All",
                label="Reading Mood",
                elem_classes=["mood-dropdown"]
            )
        search_button = gr.Button(
            "🔍 Find Your Next Read",
            elem_classes=["search-button"],
            variant="primary"
        )

    results_display = gr.HTML("", elem_classes=["results-section"])

    search_button.click(
        fn=recommend_books, 
        inputs=[description_input, category_dropdown, tone_dropdown], 
        outputs=results_display
    )

if __name__ == "__main__":
    dashboard.launch()  