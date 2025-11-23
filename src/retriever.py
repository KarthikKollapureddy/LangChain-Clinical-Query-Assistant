import os
from typing import Optional

from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings
from .waip_client import WAIPClient
import hashlib
import struct


# Simple splitter to avoid dependency on LangChain's text_splitter API
class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str):
        if not text:
            return []
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = min(start + self.chunk_size, L)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < L else end
        return chunks
from .loader import load_text_files, load_text_files_with_sources

load_dotenv()

CHROMA_DIR = os.environ.get('CHROMA_DIR', 'chroma_db')

def _get_openai_api_key() -> str:
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set. Add it to .env or export it in your environment.')
    return key

def build_vectorstore(data_dir: str, persist_dir: Optional[str] = None):
    texts_with_sources = load_text_files_with_sources(data_dir)
    splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    metadatas = []
    for relpath, text in texts_with_sources:
        chunks = splitter.split_text(text)
        for c in chunks:
            docs.append(c)
            metadatas.append({"source": relpath})

    openai_key = os.environ.get('OPENAI_API_KEY')
    dev_fake = os.environ.get('DEV_FAKE_EMBS', '') in ('1', 'true', 'True')
    waip_enabled = os.environ.get('WAIP_ENABLED', '') in ('1', 'true', 'True') or bool(os.environ.get('WAIP_API_KEY'))
    waip_client = None
    if waip_enabled and not dev_fake:
        try:
            waip_client = WAIPClient()
        except Exception:
            waip_client = None
    if not dev_fake and not waip_client and not openai_key:
        raise RuntimeError('No embedding provider configured. Set OPENAI_API_KEY, WAIP_API_KEY, or set DEV_FAKE_EMBS=1 for local dev.')

    # create chroma client
    persist_dir = persist_dir or CHROMA_DIR
    client = chromadb.Client(Settings(persist_directory=persist_dir, is_persistent=True, anonymized_telemetry=False))

    # compute embeddings (real or fake) and add to Chroma
    # If we are NOT providing embeddings (embeddings is None), delete any existing collection
    # so Chroma can recreate it and compute embeddings with the local model (avoids dimension mismatch).
    if not dev_fake and waip_client:
        # we won't provide embeddings; ensure any old collection is removed
        try:
            client.delete_collection("clinical_docs")
        except Exception:
            pass
    collection = client.get_or_create_collection(name="clinical_docs")
    embeddings = []
    if dev_fake:
        # deterministic fake embeddings via SHA256 -> floats in [-1,1]
        dim = 1536
        def text_to_emb(s: str):
            h = hashlib.sha256(s.encode('utf-8')).digest()
            vals = []
            # expand hash to required dim by repeated hashing
            cur = h
            while len(vals) < dim:
                nxt = hashlib.sha256(cur).digest()
                for i in range(0, len(nxt), 8):
                    if len(vals) >= dim:
                        break
                    chunk = nxt[i:i+8]
                    # interpret as unsigned 64-bit int and scale to [-1,1]
                    v = struct.unpack('>Q', chunk.ljust(8, b'\0'))[0]
                    vals.append((v / ((1 << 64) - 1)) * 2 - 1)
                cur = nxt
            return vals[:dim]

        for t in docs:
            embeddings.append(text_to_emb(t))
    else:
        # provider: WAIP preferred, fallback to OpenAI
        batch_size = 64
        if waip_client:
            # don't call WAIP embeddings endpoint — it may not support bulk embeddings.
            # We'll rely on a WAIP-driven generation path at query time (simple retriever).
            embeddings = None
        else:
            import openai
            openai.api_key = openai_key
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                resp = openai.Embedding.create(model="text-embedding-3-large", input=batch)
                batch_embs = [d['embedding'] for d in resp['data']]
                embeddings.extend(batch_embs)

    # add documents as items with ids (use embeddings if available)
    ids = [f"doc_{i}" for i in range(len(docs))]
    # upsert will overwrite existing IDs and avoid duplicate add/insert warnings
    if embeddings is None:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    else:
        collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    return collection

def load_vectorstore(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or CHROMA_DIR
    # Do not force an OpenAI key here; the vectorstore can be used with WAIP generation
    client = chromadb.Client(Settings(persist_directory=persist_dir, is_persistent=True, anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="clinical_docs")
    return collection

def get_qa_chain(collection, model_name: str = 'gpt-4o'):
    openai_key = os.environ.get('OPENAI_API_KEY')
    # WAIP model mapping: allow friendly names in WAIP_MODEL and map to accepted WAIP model names
    waip_model_env = os.environ.get('WAIP_MODEL')
    if waip_model_env:
        waip_model_map = {
            'gpt-4o': 'gpt-4o',
            'gpt-4': 'gpt-4',
            'gpt4': 'gpt-4',
            'gpt-35': 'gpt-35-turbo-16k',
            'gpt-35-turbo-16k': 'gpt-35-turbo-16k',
            'gpt5': 'gpt-5-chat',
        }
        model_name = waip_model_map.get(waip_model_env, waip_model_env)
    dev_fake = os.environ.get('DEV_FAKE_EMBS', '') in ('1', 'true', 'True')
    waip_enabled = os.environ.get('WAIP_ENABLED', '') in ('1', 'true', 'True') or bool(os.environ.get('WAIP_API_KEY'))
    waip_client = None
    if waip_enabled and not dev_fake:
        try:
            waip_client = WAIPClient()
        except Exception:
            waip_client = None
    if not dev_fake and not waip_client and not openai_key:
        raise RuntimeError('No generation provider configured. Set OPENAI_API_KEY, WAIP_API_KEY, or set DEV_FAKE_EMBS=1 for local dev.')

    class SimpleRetriever:
        def __init__(self, collection, k=4):
            self.collection = collection
            self.k = k

        def get_relevant_documents(self, query: str):
            # compute query embedding and use it for query
            if dev_fake:
                # same deterministic fake embedding used at index-time
                q_emb = hashlib.sha256(query.encode('utf-8')).digest()
                # build a simple numeric embedding but keep shape consistent with index dim
                # reuse small-dim fallback (approx) => use 1536 by hashing repeatedly
                dim = 1536
                vals = []
                cur = q_emb
                while len(vals) < dim:
                    nxt = hashlib.sha256(cur).digest()
                    for i in range(0, len(nxt), 8):
                        if len(vals) >= dim:
                            break
                        chunk = nxt[i:i+8]
                        v = struct.unpack('>Q', chunk.ljust(8, b'\0'))[0]
                        vals.append((v / ((1 << 64) - 1)) * 2 - 1)
                    cur = nxt
                q_emb = vals[:dim]
            else:
                if waip_client:
                    # Use Chroma text-based retrieval to get relevant docs, return docs and metadata
                    res = self.collection.query(query_texts=[query], n_results=self.k, include=['documents','metadatas','ids'])
                    docs = res.get('documents', [[]])[0]
                    mets = res.get('metadatas', [[]])[0]
                    ids = res.get('ids', [[]])[0]
                    # return list of tuples (doc_text, metadata)
                    return list(zip(docs, mets, ids))
                else:
                    import openai
                    if not openai_key:
                        raise RuntimeError('OpenAI API key not set; cannot compute query embedding')
                    openai.api_key = openai_key
                    resp = openai.Embedding.create(model="text-embedding-3-large", input=[query])
                    q_emb = resp['data'][0]['embedding']
            res = self.collection.query(query_embeddings=[q_emb], n_results=self.k, include=['documents','metadatas','ids'])
            docs = res.get('documents', [[]])[0]
            mets = res.get('metadatas', [[]])[0]
            ids = res.get('ids', [[]])[0]
            return list(zip(docs, mets, ids))

    retriever = SimpleRetriever(collection, k=4)

    def answer_query(query: str) -> dict:
        # returns {query, answer, sources: [{id, source, text_excerpt}]}
        items = retriever.get_relevant_documents(query)
        # items is list of (text, metadata, id)
        chunks = [it[0] for it in items]
        context = "\n\n".join(chunks)
        prompt = (
            f"You are a clinical assistant. Use the following context to answer the question concisely and cite sources if available:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        # If WAIP is enabled use it for generation to avoid OpenAI quota
        if waip_client is not None:
            try:
                txt = waip_client.chat_completion(prompt, model_name=model_name, max_output_tokens=512)
            except Exception as e:
                print('WAIP generation failed:', e)
                if not openai_key:
                    raise RuntimeError(f"WAIP generation failed and no OpenAI key available: {e}")
                txt = None
        else:
            txt = None

        if not txt:
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            txt = resp['choices'][0]['message']['content'].strip()

        # assemble source list
        sources = []
        for text, meta, id_ in items:
            sources.append({"id": id_, "source": meta.get('source') if isinstance(meta, dict) else meta, "text": text[:500]})

        return {"query": query, "answer": txt, "sources": sources}

    return answer_query
