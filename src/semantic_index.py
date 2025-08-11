import os, numpy as np
from typing import List, Dict, Tuple
from .models import Transaction

INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index")
os.makedirs(INDEX_DIR, exist_ok=True)

def _pack_text(t: Transaction) -> str:
    parts = [
        f"id={t.id}", f"accountId={t.account_id}", f"type={t.transaction_type}",
        f"status={t.transaction_status}", f"amount={t.amount}", f"currency={t.currency_code}",
        f"date={t.transaction_date_time}", f"merchant={t.merchant_name}"
    ]
    return "TRANSACTION " + " | ".join([p for p in parts if p and not p.endswith('=None') and not p.endswith('=')])

def build_index(transactions: List[Transaction], embed_model: str | None = None, filename: str = "tx_index"):
    from openai import OpenAI
    embed_model = embed_model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
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
    import numpy as np
    V = np.array(vecs, dtype="float32")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    path = os.path.join(INDEX_DIR, f"{filename}.npz")
    merchants = [getattr(t, "merchant_name", "") or "" for t in transactions]
categories = [getattr(t, "merchant_category_name", "") or "" for t in transactions]
np.savez_compressed(
    path,
    V=Vn,
    ids=np.array(ids),
    texts=np.array(texts),
    merchants=np.array(merchants),
    categories=np.array(categories),
    model=np.array([embed_model]),
)
    return path

def has_index(filename: str = "tx_index") -> bool:
    return os.path.exists(os.path.join(INDEX_DIR, f"{filename}.npz"))

def load_index(filename: str = "tx_index") -> Tuple[np.ndarray, list[str], list[str], list[str], list[str], str]:
    p = os.path.join(INDEX_DIR, f"{filename}.npz")
    if not os.path.exists(p): raise FileNotFoundError(p)
    data = np.load(p, allow_pickle=True)
    V = data["V"]
    ids = list(data["ids"])
    texts = list(data["texts"])
    merchants = list(data["merchants"]) if "merchants" in data.files else [""] * len(ids)
    categories = list(data["categories"]) if "categories" in data.files else [""] * len(ids)
    model = str(data["model"][0]) if "model" in data.files else ""
    return V, ids, texts, merchants, categories, model

def semantic_search(query: str, top_k: int = 12, embed_model: str | None = None, filename: str = "tx_index") -> list[dict]:
    from openai import OpenAI
    V, ids, texts, merchants, categories, _ = load_index(filename)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    embed_model = embed_model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
    qv = client.embeddings.create(model=embed_model, input=query).data[0].embedding
    import numpy as np
    q = np.array(qv, dtype="float32"); q = q/(np.linalg.norm(q)+1e-8)
    sims = V @ q
    idx = np.argsort(-sims)[:top_k]
    return [{"id": ids[i], "text": texts[i], "merchant": merchants[i], "category": categories[i], "score": float(sims[i])} for i in idx]
