import sys
sys.path.append("src")

import streamlit as st
from rag_graph import run_query

st.set_page_config(
    page_title="Advanced RAG | Self-Correction + Graph Reasoning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #111827;
        border-left: 4px solid #22c55e;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #374151;
    }
    .chunk-box {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #d1d5db;
    }
    .graph-fact {
        background-color: #0f2b1f;
        border: 1px solid #16a34a;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #86efac;
        font-family: monospace;
    }
    .stButton>button {
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #16a34a, #15803d);
    }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ How this works")
    st.markdown("""
    This system combines:
    - **Vector Search** — semantic similarity over document chunks
    - **Knowledge Graph** — entity relationships for multi-hop reasoning
    - **Self-Correction** — automatically grades and retries poor retrievals or hallucinated answers
    """)
    st.divider()
    st.markdown("### 📚 Domain")
    st.info("Renewable Energy Technologies")
    st.divider()
    st.markdown("### 💡 Try asking")
    example_queries = [
        "How does solar power compare to coal?",
        "What is a community solar project?",
        "How does energy storage support the smart grid?",
        "What factors affect the cost of solar power?"
    ]
    for eq in example_queries:
        st.markdown(f"- _{eq}_")

# ---------- HEADER ----------
st.markdown('<p class="main-header">🧠 Advanced RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Self-Correcting Retrieval + Graph-Based Multi-Hop Reasoning</p>', unsafe_allow_html=True)

# ---------- QUERY INPUT ----------
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input("", placeholder="Ask a question about renewable energy...", label_visibility="collapsed")
with col_btn:
    submit = st.button("Ask →")

# ---------- RESULTS ----------
if submit and query:
    with st.spinner("🔍 Retrieving → Grading → Reasoning → Verifying..."):
        result = run_query(query)

    st.markdown("## ✅ Answer")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    # Self-correction metrics
    st.markdown("### 🔁 Self-Correction Trace")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>{result["retries"]}</h3><p>Query Rewrites</p></div>', unsafe_allow_html=True)
    with m2:
        grounded_icon = "✅ Yes" if result["grounded"] else "⚠️ No"
        st.markdown(f'<div class="metric-card"><h3>{grounded_icon}</h3><p>Answer Grounded</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>{len(result["graph_facts"])}</h3><p>Graph Facts Used</p></div>', unsafe_allow_html=True)

    if result["query"] != query:
        st.caption(f"🔄 Final query used after self-correction: *\"{result['query']}\"*")

    st.markdown("---")

    # Two-column evidence view
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Retrieved Chunks (Vector Search)")
        if result["vec_chunks"]:
            for c in result["vec_chunks"]:
                st.markdown(f'<div class="chunk-box">{c[:280]}...</div>', unsafe_allow_html=True)
        else:
            st.caption("No chunks retrieved.")

    with col2:
        st.markdown("### 🕸️ Graph Facts (Multi-Hop Reasoning)")
        if result["graph_facts"]:
            for f in result["graph_facts"][:15]:
                st.markdown(f'<div class="graph-fact">{f}</div>', unsafe_allow_html=True)
            if len(result["graph_facts"]) > 15:
                st.caption(f"+ {len(result['graph_facts']) - 15} more facts used")
        else:
            st.caption("No graph facts matched this query.")

elif submit and not query:
    st.warning("Please enter a question first.")