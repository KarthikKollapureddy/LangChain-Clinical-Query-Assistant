import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retriever import load_vectorstore, get_qa_chain

load_dotenv()

app = FastAPI(title='LangChain Clinical Query Assistant')

class QueryRequest(BaseModel):
    query: str


@app.on_event('startup')
async def startup_event():
    global qa_chain
    try:
        # Attempt to load the vector store and create QA chain
        vectordb = load_vectorstore()
        qa_chain = get_qa_chain(vectordb)
    except Exception as e:
        qa_chain = None
        # Log the exception to stderr so users see why startup failed
        import sys
        print('Startup error:', e, file=sys.stderr)


@app.post('/query')
async def query(req: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail='QA chain not available. Build the index first or check logs.')
    try:
        answer = qa_chain.run(req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
