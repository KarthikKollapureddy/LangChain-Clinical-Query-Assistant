LangChain-Clinical-Query-Assistant
================================

Overview
--------
This project demonstrates a Retrieval-Augmented Generation (RAG) assistant for clinical queries using LangChain. It loads clinical documents, creates embeddings, stores them in a vector store (Chroma), and exposes a simple FastAPI endpoint to answer clinician queries.

Security & Privacy
------------------
- This demo uses synthetic/sample data only. Do NOT use real patient data here.
- When deploying with real EHR data, ensure encryption at rest/in transit and proper access controls.

Getting Started
---------------
1. Create a Python virtual environment and activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables (for OpenAI):

```bash
export OPENAI_API_KEY="your-openai-key"
```

3. Build the vector store (optional cached):

```bash
python src/run_demo.py --build
```

4. Run the API server:

```bash
uvicorn src.app:app --reload --port 8000
```

Usage
-----
POST `/query` with JSON `{ "query": "What's the recommended dosage for pediatric asthma?" }`

Files
-----
- `src/loader.py` - document loading utilities
- `src/retriever.py` - embedding, vector store, and QA chain
- `src/run_demo.py` - CLI to build index and run sample queries
- `src/app.py` - FastAPI server exposing `/query`

License
-------
MIT (demo)
