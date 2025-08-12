from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any
import re

from .models import Transaction
from .faiss_index import has_faiss_index, semantic_search_faiss

MONTH_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
YEAR_RE  = re.compile(r"\b(20\d{2})\b")

def _dt_key(iso: str | None) -> datetime:
    if not iso:
        return datetime.min
    try:
        return datetime.fromisoformat(iso.replace("Z","+00:00"))
    except Exception:
        try:
            return datetime.strptime(iso.split(".")[0].replace("Z",""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.min

def infer_timeframe(q: str) -> Dict[str,str]:
    q = (q or "").lower()
    m = MONTH_RE.search(q)
    if m:
        return {"month": m.group(0)}
    y = YEAR_RE.search(q)
    if y and ("this month" not in q and "last month" not in q):
        return {"year": y.group(1)}
    # leave resolution (this/last month/year) for the LLM; we won’t override here
    return {}

KEEP_FIELDS = (
    "transactionId transactionType transactionStatus transactionDateTime "
    "amount currencyCode endingBalance merchantName accountId".split()
)

def to_row_dict(t: Transaction) -> Dict[str, Any]:
    # Pydantic v2/v1 or dataclass
    d = {}
    for k in KEEP_FIELDS:
        # support snake_case too
        snake = ''.join(['_' + c.lower() if c.isupper() else c for c in k]).lstrip('_')
        v = getattr(t, snake, None)
        if v is None and hasattr(t, "model_dump"):
            v = t.model_dump(by_alias=True).get(k)
        d[k] = v
    return d

def pack_jsonl(txns: List[Transaction]) -> str:
    # compact JSONL so we can fit more rows in context
    import json
    return "\n".join(json.dumps(to_row_dict(t), separators=(",",":")) for t in txns)

def keyword_rank(query: str, txns: List[Transaction], top_k=60) -> List[Transaction]:
    q = query.lower().split()
    scored = []
    for t in txns:
        hay = f"{t.transaction_type} {t.transaction_status} {t.merchant_name} {t.currency_code} {t.account_id}".lower()
        score = sum(hay.count(term) for term in q)
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in scored[:top_k] if s > 0]

def retrieve_candidates(query: str, txns: List[Transaction], top_k=120) -> List[Transaction]:
    # 1) FAISS semantic matches (if available)
    docs = []
    if has_faiss_index("tx_faiss"):
        try:
            for d in semantic_search_faiss(query, top_k=top_k, name="tx_faiss"):
                docs.append(d["id"])
        except Exception:
            pass

    # 2) keyword
    kw = keyword_rank(query, txns, top_k=top_k)

    # 3) union & bring latest to the top
    id2t = {t.id: t for t in txns}
    pool = []
    seen = set()
    for t in kw + [id2t.get(i) for i in docs if i in id2t]:
        if not t or t.id in seen:
            continue
        seen.add(t.id)
        pool.append(t)

    pool.sort(key=lambda t: _dt_key(getattr(t, "transaction_date_time", None)), reverse=True)
    return pool[:top_k]