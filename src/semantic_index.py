# NPZ helper retained for dev fallback (not default). Not used when FAISS is present.
import os, numpy as np
from typing import List, Dict, Tuple
from .models import Transaction

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index")
os.makedirs(INDEX_DIR, exist_ok=True)

def _pack_text(t: Transaction) -> str:
    parts = [f"id={t.id}", f"accountId={t.account_id}", f"type={t.transaction_type}",
             f"status={t.transaction_status}", f"amount={t.amount}", f"currency={t.currency_code}",
             f"date={t.transaction_date_time}", f"merchant={t.merchant_name}"]
    return "TRANSACTION " + " | ".join([p for p in parts if p and not p.endswith('=None') and not p.endswith('=')])

def build_index(transactions: List[Transaction], embed_model: str | None = None, filename: str = "tx_index"):
    from openai import OpenAI
    embed_model = embed_model or os.getenv("EMBED_MODEL", "BAAI/bge-en-icl")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    texts = [_pack_text(t) for t in transactions]
    ids = [t.id for t in transactions]
    vecs = []
    chunk = 64
    for i in range(0, len(texts), chunk):
        batch = texts[i:i+chunk]
        resp = client.embeddings.create(model=embed_model, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    V = np.array(vecs, dtype="float32")
    Vn = V/(np.linalg.norm(V, axis=1, keepdims=True)+1e-8)
    path = os.path.join(INDEX_DIR, f"{filename}.npz")
    np.savez_compressed(path, V=Vn, ids=np.array(ids), texts=np.array(texts), model=np.array([embed_model]))
    return path

def has_index(filename: str = "tx_index") -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, f"{filename}.npz"))

def load_index(filename: str = "tx_index"):
    p = os.path.join(INDEX_DIR, f"{filename}.npz")
    if not os.path.exists(p): raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=True)
    return data["V"], list(data["ids"]), list(data["texts"])

def semantic_search(query: str, top_k: int = 12, embed_model: str | None = None, filename: str = "tx_index"):
    from openai import OpenAI
    V, ids, texts = load_index(filename)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    embed_model = embed_model or os.getenv("EMBED_MODEL", "BAAI/bge-en-icl")
    qv = client.embeddings.create(model=embed_model, input=query).data[0].embedding
    import numpy as np
    q = np.array(qv, dtype="float32"); q = q/(np.linalg.norm(q)+1e-8)
    sims = V @ q
    idx = np.argsort(-sims)[:top_k]
    return [{"id": ids[i], "text": texts[i], "score": float(sims[i])} for i in idx]
