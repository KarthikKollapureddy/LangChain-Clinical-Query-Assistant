import os
from typing import Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from .loader import load_text_files

load_dotenv()

CHROMA_DIR = os.environ.get('CHROMA_DIR', 'chroma_db')

def _get_openai_api_key() -> str:
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set. Add it to .env or export it in your environment.')
    return key

def build_vectorstore(data_dir: str, persist_dir: Optional[str] = None):
    texts = load_text_files(data_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for t in texts:
        docs.extend(splitter.split_text(t))

    openai_key = _get_openai_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    persist_dir = persist_dir or CHROMA_DIR
    vectordb = Chroma.from_texts(texts=docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def load_vectorstore(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or CHROMA_DIR
    openai_key = _get_openai_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def get_qa_chain(vectordb, model_name: str = 'gpt-4o-mini'):
    openai_key = _get_openai_api_key()
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_key)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa
