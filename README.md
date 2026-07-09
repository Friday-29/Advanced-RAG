# Advanced RAG Project

An Advanced Retrieval-Augmented Generation (RAG) pipeline built with Python and Streamlit to ingest, index, and query specialized documents (such as energy, carbon, and sustainability PDFs) using advanced retrieval techniques.

---

##  Features
- **Document Ingestion:** Automated processing of PDF files inside the `data/` directory.
- **Advanced Retrieval:** Enhanced chunking and retrieval strategies built with a customized graph/vector architecture.
- **Interactive UI:** Clean Streamlit web interface for conversational Q&A over your knowledge base.

---

##  Project Structure
```text
ADVANCED-RAG-PROJECT/
├── .streamlit/          # Streamlit configuration
├── data/                # Source PDF documents
├── src/                 # Source code directory
│   ├── graph_builder.py # Builds the RAG knowledge graph
│   ├── ingest.py        # Parses and ingests documents
│   ├── rag_graph.py     # Main RAG execution logic
│   └── retriever.py     # Custom retrieval mechanisms
├── .env                 # API Keys and Environment secrets (Hidden)
├── .gitignore           # Git ignore configurations
├── app.py               # Main Streamlit web application interface
└── README.md            # Project documentation