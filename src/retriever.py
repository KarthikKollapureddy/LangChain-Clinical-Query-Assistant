import os
from typing import Optional

from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings
from .waip_client import WAIPClient
import hashlib
import struct
import re


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
    # loader now returns (relpath, text, source_title)
    texts_with_sources = load_text_files_with_sources(data_dir)
    splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    metadatas = []
    # track overall and per-file chunk indices so we can recover friendly sources later
    overall_idx = 0
    for file_idx, (relpath, text, source_title) in enumerate(texts_with_sources):
        chunks = splitter.split_text(text)
        for chunk_idx, c in enumerate(chunks):
            docs.append(c)
            metadatas.append({
                "source": relpath,
                "file": relpath,
                "source_title": source_title,
                "file_index": file_idx,
                "chunk_index": chunk_idx,
                "doc_index": overall_idx,
            })
            overall_idx += 1

    openai_key = os.environ.get('OPENAI_API_KEY')
    dev_fake = os.environ.get('DEV_FAKE_EMBS', '') in ('1', 'true', 'True')
    waip_enabled = os.environ.get('WAIP_ENABLED', '') in ('1', 'true', 'True') or bool(os.environ.get('WAIP_API_KEY'))
    use_st = os.environ.get('USE_SENTENCE_TRANSFORMERS', '') in ('1', 'true', 'True')
    st_model_name = os.environ.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    waip_client = None
    if waip_enabled and not dev_fake:
        try:
            waip_client = WAIPClient()
        except Exception:
            waip_client = None
    # Require at least one embedding provider: prefer OpenAI when available
    if not dev_fake and not openai_key and not waip_client:
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
        # provider selection order: sentence-transformers (local) -> OpenAI -> WAIP
        batch_size = 64
        if use_st:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError(f"Sentence-Transformers requested but not installed: {e}")
            model = SentenceTransformer(st_model_name)
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                embs = model.encode(batch, convert_to_numpy=True)
                embeddings.extend([e.astype(float).tolist() for e in embs])
        elif openai_key:
            import openai
            openai.api_key = openai_key
            for i in range(0, len(docs), batch_size):
                batch = docs[i : i + batch_size]
                resp = openai.Embedding.create(model="text-embedding-3-large", input=batch)
                batch_embs = [d['embedding'] for d in resp['data']]
                embeddings.extend(batch_embs)
        elif waip_client:
            # don't call WAIP embeddings endpoint — it may not support bulk embeddings.
            # We'll rely on a WAIP-driven generation path at query time (simple retriever).
            embeddings = None
        else:
            raise RuntimeError('No embedding provider available for building vectorstore')

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
                    # Chroma may not support returning 'ids' via include in all versions.
                    # Request only supported fields and synthesize ids from the returned documents/metadata.
                    res = self.collection.query(query_texts=[query], n_results=self.k, include=['documents','metadatas'])
                    docs = res.get('documents', [[]])[0]
                    mets = res.get('metadatas', [[]])[0]
                    # synthesize stable ids based on the metadata 'source' and document snippet index
                    ids = []
                    for i, (m, d) in enumerate(zip(mets, docs)):
                        # prefer metadata source when available, otherwise synthesize from content
                        if isinstance(m, dict) and (m.get('source_title') or m.get('source')):
                            src = m.get('source_title') or m.get('source')
                        elif m:
                            src = str(m)
                        else:
                            # fallback: short hash of the document text for stable id/reference
                            h = hashlib.sha256(d.encode('utf-8')).hexdigest()[:8]
                            src = f"doc_{h}"
                        ids.append(f"{src}::chunk_{i}")
                    return list(zip(docs, mets, ids))
                else:
                    import openai
                    if not openai_key:
                        raise RuntimeError('OpenAI API key not set; cannot compute query embedding')
                    openai.api_key = openai_key
                    resp = openai.Embedding.create(model="text-embedding-3-large", input=[query])
                    q_emb = resp['data'][0]['embedding']
            res = self.collection.query(query_embeddings=[q_emb], n_results=self.k, include=['documents','metadatas'])
            docs = res.get('documents', [[]])[0]
            mets = res.get('metadatas', [[]])[0]
            ids = []
            for i, (m, d) in enumerate(zip(mets, docs)):
                if isinstance(m, dict) and (m.get('source_title') or m.get('source')):
                    src = m.get('source_title') or m.get('source')
                elif m:
                    src = str(m)
                else:
                    h = hashlib.sha256(d.encode('utf-8')).hexdigest()[:8]
                    src = f"doc_{h}"
                ids.append(f"{src}::chunk_{i}")
            return list(zip(docs, mets, ids))

    retriever = SimpleRetriever(collection, k=4)

    def answer_query(query: str) -> dict:
        # return {query, answer, sources: [{id, source, text_excerpt}]}
        items = retriever.get_relevant_documents(query)
        # items is list of (text, metadata, id)
        chunks = [it[0] for it in items]
        context = "\n\n".join(chunks)
        # If any retrieved item lacks a clear source label, instruct the model to provide citations
        missing_source = any(not (isinstance(it[1], dict) and (it[1].get('source_title') or it[1].get('source'))) for it in items)
        prompt = (
            f"You are a clinical assistant. Use the following context to answer the question concisely and cite sources if available:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        if missing_source:
            prompt += (
                "\n\nNote: Some context chunks do not include source labels. When answering, try to identify and list the source for any facts you rely on. "
                "If you cannot identify a precise source from the provided context, explicitly mark that source as 'unknown'. Provide inline citations or a short 'Sources:' list after the answer."
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

        # If the model appended a 'Sources' block in the generated text, extract it
        model_sources = []
        # match a trailing 'Sources:' or 'Sources' line and grab what's after it
        m = re.search(r"\bSources\b\s*:?(.*)$", txt, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sources_text = m.group(1).strip()
            # remove the matched block from the answer text
            txt = txt[:m.start()].strip()
            # split into lines/entries by newlines or commas
            parts = [s.strip(' -\t\r\n') for s in re.split(r"[\r\n]+|,\s*", sources_text) if s.strip()]
            model_sources = parts

        # assemble retriever source list: prefer semantic similarity matching
        # If sentence-transformers is available, embed the final answer and each chunk
        # and include any source where cosine similarity >= threshold.
        sources_set = []
        st_available = False
        st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            from numpy import dot
            from numpy.linalg import norm
            import numpy as np
            st_available = True
        except Exception:
            st_available = False

        if st_available:
            try:
                st_model_name = os.environ.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
                st_model = SentenceTransformer(st_model_name)
                # compute answer embedding and chunk embeddings
                ans_emb = st_model.encode([txt], convert_to_numpy=True)[0].astype(float)
                chunks = [ (it[0] or '')[:1000] for it in items ]
                if chunks:
                    ch_embs = st_model.encode(chunks, convert_to_numpy=True)
                    # cosine similarity and thresholding
                    threshold = float(os.environ.get('SOURCE_SIMILARITY_THRESHOLD', 0.7))
                    for (text, meta, id_), ch_e in zip(items, ch_embs):
                        sim = float(np.dot(ans_emb, ch_e) / (np.linalg.norm(ans_emb) * np.linalg.norm(ch_e) + 1e-12))
                        if sim >= threshold:
                            if isinstance(meta, dict):
                                src_label = meta.get('source_title') or meta.get('source') or meta.get('file')
                            else:
                                src_label = str(meta)
                            src_label = src_label or 'unknown'
                            if src_label not in sources_set:
                                sources_set.append(src_label)
            except Exception:
                # If semantic matching failed, fall back to substring heuristic below
                st_available = False

        if not st_available:
            # fallback: substring/excerpt matching (cheap)
            answer_lower = txt.lower()
            for text, meta, id_ in items:
                if isinstance(meta, dict):
                    src_label = meta.get('source_title') or meta.get('source') or meta.get('file')
                else:
                    src_label = str(meta)
                src_label = src_label or 'unknown'
                excerpt = (text or '').strip()
                if not excerpt:
                    continue
                excerpt = excerpt[:200]
                if excerpt.lower() in answer_lower:
                    if src_label not in sources_set:
                        sources_set.append(src_label)

        sources = sources_set

        # If the model reported its own sources, prefer those (they are authoritative)
        if model_sources:
            # ensure unique and preserve order
            seen = set()
            picked = []
            for s in model_sources:
                if s not in seen:
                    seen.add(s)
                    picked.append(s)
            sources = picked

        # If still empty, fall back to the retriever-provided source labels (deduped in-order)
        if not sources:
            fallback = []
            seen = set()
            for text, meta, id_ in items:
                if isinstance(meta, dict):
                    src_label = meta.get('source_title') or meta.get('source') or meta.get('file')
                else:
                    src_label = str(meta)
                src_label = src_label or 'unknown'
                if src_label not in seen:
                    seen.add(src_label)
                    fallback.append(src_label)
            sources = fallback

        # Return model-cleaned answer, retriever sources, and any sources the model reported
        return {"query": query, "answer": txt, "sources": sources, "model_sources": model_sources}

    return answer_query
