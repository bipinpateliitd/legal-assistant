# Legal Assistant

## Environment Setup

```bash
# Initialize uv
uv init

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add -r requirements.txt
```

This project uses `uv` for Python package management and virtual environment handling. The commands above will set up the environment and install all required dependencies.

## Running Scripts

### Data Collection

```bash
# Run the data scraping script
uv run web_scrap_cleaned_data.py

# Run the metadata scraping script
uv run web_scrap_cleaned_metadata.py

# Run the keyword extraction script
uv run top_keywords.py
```

The `top_keywords.py` script extracts the most relevant legal keywords from each section of the Bhartiya Nyaya Sanhita.

### Vector Database and RAG

```bash
# Calculate embeddings, chunk documents, and save to Qdrant vector database
uv run ingest_bns_embedding_qdrant_hybrid.py

# Run RAG evaluation
uv run rag_evaluation.py
```

The `ingest_bns_embedding_qdrant_hybrid.py` script processes documents by chunking them and calculating embeddings, then stores them in a Qdrant vector database for efficient retrieval. The `rag_evaluation.py` script evaluates the performance of the Retrieval-Augmented Generation system.

### Running the Application

```bash
# Launch the Streamlit web application
streamlit run app.py
```

This will start the Streamlit web interface for the legal assistant, allowing users to interact with the system through a user-friendly interface.