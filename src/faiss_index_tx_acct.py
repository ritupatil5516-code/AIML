from __future__ import annotations

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Iterable, Tuple, Optional

# Embeddings: you can route via LiteLLM/OpenAI or local model.
# Here we default to sentence-transformers BGE via sentence_transformers.
# Install once: pip install sentence-transformers faiss-cpu
from sentence_transformers import SentenceTransformer

_DEFAULT_EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_INDEX_DIR = os.getenv("FAISS_DIR", "indexes")

# --------------- utils ---------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_model(name: str) -> SentenceTransformer:
    # Keep a global cache so Streamlit reruns don't reload every time
    global _EMB_MODEL_CACHE
    try:
        cache = _EMB_MODEL_CACHE
    except NameError:
        _EMB_MODEL_CACHE = {}
        cache = _EMB_MODEL_CACHE
    if name not in cache:
        cache[name] = SentenceTransformer(name)
    return cache[name]

def _normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (int, float)):
        return str(s)
    if isinstance(s, (list, tuple)):
        return ", ".join(map(_normalize_text, s))
    return str(s)

# --------------- packers (keep in sync with retrieval) ---------------

_TX_FIELDS = (
    "transactionId transactionType transactionStatus transactionDateTime "
    "amount currencyCode endingBalance merchantName accountId".split()
)

def pack_tx_text(d: Dict[str, Any]) -> str:
    # `d` can be a Pydantic model-dump or dict
    parts = []
    for k in _TX_FIELDS:
        v = d.get(k)
        if v is None:
            continue
        parts.append(f"{k}:{_normalize_text(v)}")
    return " | ".join(parts)

_ACCT_FIELDS = [
    "accountId","accountNumberLast4","accountStatus","accountType","productType",
    "currentBalance","currentAdjustedBalance","totalBalance","availableCredit","creditLimit",
    "minimumDueAmount","pastDueAmount","highestPriorityStatus","balanceStatus",
    "paymentDueDate","paymentDueDateTime","billingCycleOpenDateTime","billingCycleCloseDateTime",
    "lastUpdatedDate","openedDate","closedDate","currencyCode","flags","subStatuses"
]

def pack_acct_text(d: Dict[str, Any]) -> str:
    parts = []
    for k in _ACCT_FIELDS:
        v = d.get(k)
        if v is None:
            continue
        parts.append(f"{k}:{_normalize_text(v)}")
    return " | ".join(parts)

# --------------- core FAISS I/O ---------------

def _save_faiss(index: faiss.Index, meta: Dict[str, Any], name: str):
    _ensure_dir(_INDEX_DIR)
    faiss_path = os.path.join(_INDEX_DIR, f"{name}.faiss")
    meta_path  = os.path.join(_INDEX_DIR, f"{name}.meta.json")
    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def _load_faiss(name: str) -> Tuple[faiss.Index, Dict[str, Any]]:
    faiss_path = os.path.join(_INDEX_DIR, f"{name}.faiss")
    meta_path  = os.path.join(_INDEX_DIR, f"{name}.meta.json")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"FAISS index '{name}' not found in {_INDEX_DIR}")
    index = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def has_faiss_index(name: str) -> bool:
    return os.path.exists(os.path.join(_INDEX_DIR, f"{name}.faiss")) and \
           os.path.exists(os.path.join(_INDEX_DIR, f"{name}.meta.json"))

# --------------- build helpers ---------------

def _embed_texts(texts: List[str], model_name: str | None = None) -> np.ndarray:
    model = _load_model(model_name or _DEFAULT_EMBEDDING_MODEL)
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)

def _build_flat_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)          # cosine if vectors are normalized
    index.add(vectors)
    return index

# --------------- PUBLIC: build indexes ---------------

def build_tx_index(rows: List[Dict[str, Any]], *, name: str = "tx_faiss",
                   model_name: str | None = None) -> None:
    """
    rows: list of dicts with at least keys:
          transactionId, transactionType, transactionStatus, transactionDateTime, amount, ...
    """
    ids: List[str] = []
    texts: List[str] = []
    for r in rows:
        tid = r.get("transactionId") or r.get("id")
        if not tid:
            # skip rows without stable id
            continue
        ids.append(str(tid))
        texts.append(pack_tx_text(r))

    if not ids:
        raise ValueError("No transaction rows with IDs to index.")

    vecs = _embed_texts(texts, model_name=model_name)
    index = _build_flat_index(vecs)

    meta = {
        "name": name,
        "type": "transactions",
        "model": model_name or _DEFAULT_EMBEDDING_MODEL,
        "ids": ids,
        "fields": _TX_FIELDS,
    }
    _save_faiss(index, meta, name)

def build_account_index(rows: List[Dict[str, Any]], *, name: str = "acct_faiss",
                        model_name: str | None = None) -> None:
    """
    rows: list of dicts with at least keys:
          accountId and account summary fields
    """
    ids: List[str] = []
    texts: List[str] = []
    for r in rows:
        aid = r.get("accountId") or r.get("id")
        if not aid:
            continue
        ids.append(str(aid))
        texts.append(pack_acct_text(r))

    if not ids:
        raise ValueError("No account rows with IDs to index.")

    vecs = _embed_texts(texts, model_name=model_name)
    index = _build_flat_index(vecs)

    meta = {
        "name": name,
        "type": "accounts",
        "model": model_name or _DEFAULT_EMBEDDING_MODEL,
        "ids": ids,
        "fields": _ACCT_FIELDS,
    }
    _save_faiss(index, meta, name)

# --------------- PUBLIC: search ---------------

def semantic_search_faiss(query: str, *, top_k: int = 20, name: str) -> List[Dict[str, Any]]:
    """
    Returns list of {id: str, score: float, rank: int}
    """
    index, meta = _load_faiss(name)
    model = _load_model(meta["model"])
    qv = model.encode([query], normalize_embeddings=True)
    qv = np.asarray(qv, dtype=np.float32)
    scores, idxs = index.search(qv, k=min(top_k, index.ntotal))
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    ids = meta["ids"]
    out = []
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        if i < 0:
            continue
        out.append({"id": ids[i], "score": float(s), "rank": rank})
    return out