import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retriever import load_vectorstore, get_qa_chain

app = FastAPI(title='LangChain Clinical Query Assistant')

class QueryRequest(BaseModel):
    query: str

@app.on_event('startup')
async def startup_event():
    global qa_chain
    try:
        vectordb = load_vectorstore()
        qa_chain = get_qa_chain(vectordb)
    except Exception as e:
        qa_chain = None

@app.post('/query')
async def query(req: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail='QA chain not available. Build the index first or check logs.')
    try:
        answer = qa_chain.run(req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
