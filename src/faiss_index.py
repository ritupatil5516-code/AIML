import os, json, numpy as np
from typing import List, Dict
from .models import Transaction

try:
    import faiss  # faiss-cpu or faiss-gpu
except Exception as e:
    raise RuntimeError("faiss is required. Install with `pip install faiss-cpu`.") from e

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_faiss")
os.makedirs(INDEX_DIR, exist_ok=True)

def _pack_text(t: Transaction) -> str:
    parts = [
        f"id={t.id}", f"accountId={t.account_id}", f"type={t.transaction_type}",
        f"status={t.transaction_status}", f"amount={t.amount}", f"currency={t.currency_code}",
        f"date={t.transaction_date_time}", f"merchant={t.merchant_name}"
    ]
    return "TRANSACTION " + " | ".join([p for p in parts if p and not p.endswith('=None') and not p.endswith('=')])

def _embed_texts(texts: List[str], embed_model: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    vecs = []
    chunk = 64
    for i in range(0, len(texts), chunk):
        batch = texts[i:i+chunk]
        resp = client.embeddings.create(model=embed_model, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    V = np.array(vecs, dtype="float32")
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)  # L2 normalize
    return V

def build_faiss_index(transactions: List[Transaction], embed_model: str | None = None, name: str = "tx_faiss"):
    embed_model = embed_model or os.getenv("EMBED_MODEL", "BAAI/bge-en-icl")
    texts = [_pack_text(t) for t in transactions]
    ids = [t.id for t in transactions]
    merchants = [t.merchant_name or "" for t in transactions]
    categories = [getattr(t, "merchant_category_name", None) or "" for t in transactions]

    V = _embed_texts(texts, embed_model)
    dim = V.shape[1]

    # Exact cosine via inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(V)

    idx_path = os.path.join(INDEX_DIR, f"{name}.index")
    faiss.write_index(index, idx_path)

    meta = {"ids": ids, "texts": texts, "merchants": merchants, "categories": categories, "model": embed_model, "dim": dim}
    meta_path = os.path.join(INDEX_DIR, f"{name}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return idx_path, meta_path

def has_faiss_index(name: str = "tx_faiss") -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, f"{name}.index")) and os.path.exists(os.path.join(INDEX_DIR, f"{name}.meta.json"))

def _load_index_and_meta(name: str = "tx_faiss"):
    idx_path = os.path.join(INDEX_DIR, f"{name}.index")
    meta_path = os.path.join(INDEX_DIR, f"{name}.meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("FAISS index or metadata not found")
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def semantic_search_faiss(query: str, top_k: int = 12, embed_model: str | None = None, name: str = "tx_faiss") -> List[Dict[str, str]]:
    embed_model = embed_model or os.getenv("EMBED_MODEL", "BAAI/bge-en-icl")
    index, meta = _load_index_and_meta(name)

    # Embed query
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    qv = client.embeddings.create(model=embed_model, input=query).data[0].embedding
    import numpy as np
    q = np.array(qv, dtype="float32"); q = q/(np.linalg.norm(q)+1e-8)

    sims, idxs = index.search(q.reshape(1,-1), top_k)
    sims = sims[0]; idxs = idxs[0]

    docs = []
    for score, i in zip(sims, idxs):
        if i < 0: continue
        docs.append({
            "id": meta["ids"][i],
            "text": meta["texts"][i],
            "merchant": meta["merchants"][i],
            "category": meta["categories"][i],
            "score": float(score)
        })
    return docs
