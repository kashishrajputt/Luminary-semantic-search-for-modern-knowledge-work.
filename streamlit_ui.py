"""
Luminary - Semantic search for modern knowledge work
Premium neumorphic design with soft lavender aesthetics
"""
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from src.cache_manager import CacheManager
from src.embedder import EmbeddingGenerator
from src.preprocess import preprocess_documents
from src.search_engine import QueryExpander, SearchEngine
from src.utils import Config


# =============================================================================
# Configuration
# =============================================================================
BRAND_NAME = "Luminary"
BRAND_TAGLINE = "Semantic search for modern knowledge work"
LOGO_PATH = "Gemini_Generated_Image_xsluu1xsluu1xslu.png"  # Update with your logo path

# Color Palette
BG_GRADIENT_START = "#ECEBFF"
BG_GRADIENT_END = "#F3F2FF"
CARD_BG = "#F7F6FF"
PRIMARY_LAVENDER = "#8B8EF5"
SOFT_LAVENDER = "#C7C9FF"
TEXT_COLOR = "#22242F"
MUTED_TEXT = "#6B7280"
BORDER_COLOR = "#E2E4FF"


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title=f"{BRAND_NAME} - Semantic Search",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Custom CSS - Neumorphic Design + FIXES
# =============================================================================
st.markdown(f"""
<style>
    /* Import SF Pro-like font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Main background gradient */
    .main {{
        background: linear-gradient(135deg, {BG_GRADIENT_START} 0%, {BG_GRADIENT_END} 100%);
    }}
    
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    
    /* Header Card - Neumorphic */
    .header-card {{
        background: {CARD_BG};
        padding: 1.5rem 2rem;
        border-radius: 26px;
        margin-bottom: 2.5rem;
        box-shadow: 
            -6px -6px 16px rgba(255, 255, 255, 0.8),
            6px 6px 16px rgba(199, 201, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    
    .header-content {{
        display: flex;
        align-items: center;
        gap: 1rem;
        width: 100%;
    }}
    
    .header-text {{
        flex: 1;
    }}
    
    .brand-title {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }}
    
    .brand-tagline {{
        font-size: 0.95rem;
        color: {MUTED_TEXT};
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CARD_BG} 0%, {BG_GRADIENT_END} 100%);
    }}
    
    section[data-testid="stSidebar"] > div {{
        padding: 2rem 1.5rem;
    }}
    
    /* Navigation Pills */
    .nav-pill {{
        background: {CARD_BG};
        padding: 0.75rem 1.25rem;
        border-radius: 18px;
        margin-bottom: 0.75rem;
        color: {MUTED_TEXT};
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 
            -4px -4px 12px rgba(255, 255, 255, 0.7),
            4px 4px 12px rgba(199, 201, 255, 0.25);
    }}
    
    .nav-pill.active {{
        background: linear-gradient(135deg, {PRIMARY_LAVENDER} 0%, {SOFT_LAVENDER} 100%);
        color: white;
        box-shadow: 
            -2px -2px 8px rgba(255, 255, 255, 0.5),
            4px 4px 16px rgba(139, 142, 245, 0.4);
    }}
    
    .nav-pill:hover {{
        transform: translateY(-2px);
    }}
    
    /* Search Input - FIXED (no double purple bar) */
    /* Outer container of the text input */
    .stTextInput > div {{
        padding: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    
    /* Capsule wrapper */
    .stTextInput > div > div {{
        background: {CARD_BG} !important;
        border-radius: 24px !important;
        border: 1px solid {BORDER_COLOR} !important;
        box-shadow:
            inset -2px -2px 8px rgba(255, 255, 255, 0.8),
            inset 2px 2px 8px rgba(199, 201, 255, 0.2);
        padding: 0.1rem 1.25rem !important;
    }}
    
    /* Only one visible border; input itself is transparent */
    .stTextInput > div > div > input {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.6rem 0 0.6rem 0 !important;
        font-size: 1rem;
        color: {TEXT_COLOR};
    }}
    
    .stTextInput > div > div:focus-within {{
        border-color: {PRIMARY_LAVENDER} !important;
        box-shadow:
            inset -2px -2px 8px rgba(255, 255, 255, 0.8),
            inset 2px 2px 8px rgba(199, 201, 255, 0.3),
            0 0 0 3px rgba(139,142,245,0.15) !important;
    }}
    
    /* Buttons - Neumorphic Gradient */
    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY_LAVENDER} 0%, {SOFT_LAVENDER} 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: 24px;
        font-size: 1rem;
        box-shadow: 
            -4px -4px 12px rgba(255, 255, 255, 0.6),
            4px 4px 12px rgba(139, 142, 245, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 0.01em;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            -4px -4px 16px rgba(255, 255, 255, 0.7),
            6px 6px 20px rgba(139, 142, 245, 0.5);
    }}
    
    .stButton > button:active {{
        transform: translateY(0px);
    }}
    
    /* Stat Cards - Neumorphic */
    .stat-card {{
        background: {CARD_BG};
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 
            -6px -6px 14px rgba(255, 255, 255, 0.8),
            6px 6px 14px rgba(199, 201, 255, 0.3);
        transition: transform 0.3s ease;
    }}
    
    .stat-card:hover {{
        transform: translateY(-4px);
    }}
    
    .stat-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {PRIMARY_LAVENDER};
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}
    
    .stat-label {{
        font-size: 0.875rem;
        color: {MUTED_TEXT};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    /* Result Cards */
    .result-card {{
        background: {CARD_BG};
        padding: 1.75rem;
        border-radius: 22px;
        margin-bottom: 1.25rem;
        box-shadow: 
            -6px -6px 16px rgba(255, 255, 255, 0.8),
            6px 6px 16px rgba(199, 201, 255, 0.3);
        transition: all 0.3s ease;
        border-left: 4px solid {PRIMARY_LAVENDER};
    }}
    
    .result-card:hover {{
        transform: translateX(6px);
        box-shadow: 
            -8px -8px 20px rgba(255, 255, 255, 0.85),
            8px 8px 20px rgba(199, 201, 255, 0.35);
    }}
    
    .result-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .result-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        letter-spacing: -0.01em;
    }}
    
    .result-score {{
        background: linear-gradient(135deg, {PRIMARY_LAVENDER} 0%, {SOFT_LAVENDER} 100%);
        color: white;
        padding: 0.375rem 1rem;
        border-radius: 16px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 
            -2px -2px 6px rgba(255, 255, 255, 0.4),
            2px 2px 6px rgba(139, 142, 245, 0.3);
    }}
    
    .result-preview {{
        color: {MUTED_TEXT};
        line-height: 1.65;
        margin: 1rem 0;
        font-size: 0.95rem;
    }}
    
    .result-meta {{
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER_COLOR};
    }}
    
    .meta-item {{
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }}
    
    .meta-label {{
        font-size: 0.75rem;
        color: {MUTED_TEXT};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    
    .meta-value {{
        font-size: 0.95rem;
        color: {TEXT_COLOR};
        font-weight: 600;
    }}
    
    /* Keyword Pills */
    .keyword-pill {{
        display: inline-block;
        background: linear-gradient(135deg, {SOFT_LAVENDER}40 0%, {PRIMARY_LAVENDER}20 100%);
        color: {PRIMARY_LAVENDER};
        padding: 0.375rem 0.875rem;
        border-radius: 14px;
        margin: 0.25rem;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 
            -2px -2px 6px rgba(255, 255, 255, 0.6),
            2px 2px 6px rgba(199, 201, 255, 0.2);
    }}
    
    /* Search Bar Container */
    .search-container {{
        background: {CARD_BG};
        padding: 1.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 
            -6px -6px 16px rgba(255, 255, 255, 0.8),
            6px 6px 16px rgba(199, 201, 255, 0.3);
    }}
    
    /* Empty State */
    .empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        color: {MUTED_TEXT};
    }}
    
    .empty-state h3 {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin-bottom: 0.75rem;
    }}
    
    .empty-state p {{
        font-size: 1.05rem;
        color: {MUTED_TEXT};
    }}
    
    /* Sliders */
    .stSlider > div > div > div {{
        background: {PRIMARY_LAVENDER};
    }}
    
    /* Checkboxes */
    .stCheckbox > label {{
        color: {TEXT_COLOR};
        font-weight: 500;
    }}
    
    /* Hide Streamlit branding but KEEP header visible for sidebar toggle */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{
        visibility: visible !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.01em;
    }}
    
    /* Info message styling */
    .stAlert {{
        background: {CARD_BG};
        border-radius: 18px;
        border-left: 4px solid {PRIMARY_LAVENDER};
        box-shadow: 
            -4px -4px 12px rgba(255, 255, 255, 0.7),
            4px 4px 12px rgba(199, 201, 255, 0.25);
    }}
    
    /* --- FIX: Always show sidebar toggle button --- */
    [data-testid="collapsedControl"],
    button[kind="iconButton"],
    button[kind="sidebar-toggle"],
    button[title="Toggle sidebar"] {{
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
        z-index: 9999 !important;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cache Engine Components
# =============================================================================
@st.cache_resource
def load_engine():
    """Load and cache the search engine components."""
    Config.init_dirs()
    
    cache = CacheManager()
    embedder = EmbeddingGenerator(cache_manager=cache)
    engine = SearchEngine(dimension=embedder.embedding_dim)
    expander = QueryExpander()
    
    docs = preprocess_documents(Config.DATA_DIR)
    if docs:
        embeddings = embedder.embed_documents(docs)
        engine.index_documents(docs, embeddings)
    
    return embedder, engine, expander, cache


# =============================================================================
# UI Components
# =============================================================================
def render_header():
    """Render neumorphic header with logo."""
    logo_path = Path(LOGO_PATH)
    
    if logo_path.exists():
        col1, col2 = st.columns([0.5, 11.5])
        with col1:
            st.image(str(logo_path), width=42)
        with col2:
            st.markdown(f"""
            <div style="margin-top: -10px;">
                <div class="brand-title">{BRAND_NAME}</div>
                <div class="brand-tagline">{BRAND_TAGLINE}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([0.5, 11.5])
        with col1:
            st.markdown(f"""
            <div style="
                width: 42px;
                height: 42px;
                background: linear-gradient(135deg, {PRIMARY_LAVENDER} 0%, {SOFT_LAVENDER} 100%);
                border-radius: 12px;
                box-shadow: -2px -2px 6px rgba(255,255,255,0.6), 2px 2px 6px rgba(199,201,255,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 700;
                font-size: 1.25rem;
            ">L</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="margin-top: -10px;">
                <div class="brand-title">{BRAND_NAME}</div>
                <div class="brand-tagline">{BRAND_TAGLINE}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    [data-testid="column"]:first-child {
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)


def render_stats(engine, cache):
    """Render statistics cards."""
    stats = cache.get_stats()
    index_type = "FAISS" if hasattr(engine.index, "index") else "NumPy"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{engine.document_count}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{stats["total_entries"]}</div>
            <div class="stat-label">Cache Count</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{index_type}</div>
            <div class="stat-label">Index Type</div>
        </div>
        """, unsafe_allow_html=True)


def render_result_card(result, index):
    """Render a single result card."""
    keywords = result.explanation.get("keywords", [])[:6]
    keywords_html = "".join([f'<span class="keyword-pill">{kw}</span>' for kw in keywords])
    
    overlap = result.explanation.get("overlap_ratio", 0)
    doc_length = result.explanation.get("doc_length", 0)
    
    st.markdown(f"""
    <div class="result-card">
        <div class="result-header">
            <div class="result-title">{result.doc_id}</div>
            <div class="result-score">{result.score:.1%} Match</div>
        </div>
        <div class="result-preview">{result.preview}</div>
        <div style="margin: 1rem 0;">
            {keywords_html if keywords_html else '<span style="color: #9ca3af; font-size: 0.875rem;">No matching keywords</span>'}
        </div>
        <div class="result-meta">
            <div class="meta-item">
                <div class="meta-label">Overlap</div>
                <div class="meta-value">{overlap:.1%}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Length</div>
                <div class="meta-value">{doc_length}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_score_chart(results):
    """Create score distribution chart."""
    if not results:
        return None
    
    scores = [r.score for r in results]
    doc_ids = [r.doc_id[:25] for r in results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=doc_ids,
            y=scores,
            marker=dict(
                color=scores,
                colorscale=[[0, SOFT_LAVENDER], [1, PRIMARY_LAVENDER]],
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ),
            text=[f"{s:.1%}" for s in scores],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Score: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Score Distribution",
            font=dict(size=18, color=TEXT_COLOR, family="Inter")
        ),
        xaxis_title="",
        yaxis_title="Similarity Score",
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color=MUTED_TEXT),
        margin=dict(t=60, b=60, l=60, r=20),
        yaxis=dict(gridcolor='rgba(199, 201, 255, 0.2)')
    )
    
    return fig


# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main application entry point."""
    
    render_header()
    
    with st.spinner("Initializing search engine..."):
        embedder, engine, expander, cache = load_engine()
    
    # Sidebar
    with st.sidebar:
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 'search'
        
        if st.button("Search", use_container_width=True, key="nav_search"):
            st.session_state.current_tab = 'search'
        
        if st.button("Documents", use_container_width=True, key="nav_docs"):
            st.session_state.current_tab = 'documents'
        
        if st.button("Settings", use_container_width=True, key="nav_settings"):
            st.session_state.current_tab = 'settings'
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Search Settings</div>', unsafe_allow_html=True)
        
        top_k = st.slider("Number of results", 1, 20, 5, help="Results to return")
        expand_query = st.checkbox("Query expansion", False, help="Add synonyms using WordNet")
        length_normalize = st.checkbox("Length normalization", False, help="Adjust scores by document length")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.button("Reindex Documents", use_container_width=True, key="reindex_btn"):
            st.cache_resource.clear()
            st.rerun()
    
    # Main Content
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Search",
            placeholder="Enter your search query...",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("Search", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Search Logic
    if query and (search_button or query):
        with st.spinner("Searching..."):
            search_query = query
            if expand_query:
                search_query = expander.expand(query)
                if search_query != query:
                    st.info(f"Expanded query: `{search_query}`")
            
            query_embedding = embedder.embed_query(search_query)
            response = engine.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                include_explanation=True,
                length_normalize=length_normalize
            )
        
        if response.results:
            st.markdown(f"""
            <div style="margin: 1.5rem 0; color: {TEXT_COLOR}; font-weight: 600;">
                Found {len(response.results)} results in {response.search_time_ms:.2f}ms
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
            for i, result in enumerate(response.results, 1):
                render_result_card(result, i)
            
            st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
            render_stats(engine, cache)
            
            st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)
            fig = create_score_chart(response.results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No results found. Try a different query.")
            render_stats(engine, cache)
    
    else:
        st.markdown("""
        <div class="empty-state">
            <h3>Start by entering a query above</h3>
            <p>Search through your knowledge base using natural language</p>
        </div>
        """, unsafe_allow_html=True)
        
        render_stats(engine, cache)


if __name__ == "__main__":
    main()
