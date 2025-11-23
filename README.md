LangChain-Clinical-Query-Assistant
================================

[![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)

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

You can either export `OPENAI_API_KEY` in your shell or place it in a `.env` file at the project root. The code uses `python-dotenv` to automatically load `.env`.

Shell example:

```bash
export OPENAI_API_KEY="your-openai-key"
```

Or create `.env` with:

```dotenv
OPENAI_API_KEY=your-openai-key
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

Run & Test (local)
------------------
1. Create and activate the virtualenv (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Add credentials to `.env` in the project root. Examples (choose one provider):

OpenAI (embeddings + generation):

```env
OPENAI_API_KEY=sk-...
```

WAIP (use as alternative LLM for generation):

```env
WAIP_API_KEY=token|...
WAIP_API_ENDPOINT=https://api.waip.wiprocms.com
WAIP_ENABLED=1
```

3. Build the vector store from `data/` (optional but recommended):

```bash
python src/run_demo.py --build --data data
```

4. Start the API server (development):

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

5. Test the endpoint with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" \
	-d '{"query":"What are common side effects of aspirin?"}'
```

Troubleshooting
---------------
- If you see OpenAI quota or key errors, either set `OPENAI_API_KEY` correctly or enable `WAIP` in `.env`.
- If WAIP returns HTTP 422, check `WAIP_API_ENDPOINT` and `WAIP_API_KEY` values; some WAIP deployments accept only specific `model_name` values — try setting `WAIP_MODEL=gpt-4o` or another model listed in the WAIP validation response.

Files
-----
- `src/loader.py` - document loading utilities
- `src/retriever.py` - embedding, vector store, and QA chain
- `src/run_demo.py` - CLI to build index and run sample queries
- `src/app.py` - FastAPI server exposing `/query`

License
-------
MIT (demo)

Sample data
-----------
The `data/` directory contains `sample_clinical.txt` with example guideline excerpts used for local testing. If you add or modify files under `data/`, rebuild the vector store so the new content is indexed:

```bash
python src/run_demo.py --build --data data
```

Commiting changes
-----------------
After modifying sample data or code, you can commit the updates:

```bash
git add -A
git commit -m "Update sample clinical data and README"
```
