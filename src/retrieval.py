import os
from typing import List, Dict
from .models import Transaction
from .semantic_index import has_index, semantic_search
from .faiss_index import has_faiss_index, semantic_search_faiss

def _pack_text(t: Transaction) -> str:
    parts = [f"id={t.id}", f"accountId={t.account_id}", f"type={t.transaction_type}",
             f"status={t.transaction_status}", f"amount={t.amount}", f"currency={t.currency_code}",
             f"date={t.transaction_date_time}", f"merchant={t.merchant_name}"]
    return "TRANSACTION " + " | ".join([p for p in parts if p and not p.endswith('=None') and not p.endswith('=')])

def _keyword_rank(query: str, docs: List[Dict[str,str]], top_k=12):
    q = query.lower().split()
    scored = []
    for d in docs:
        text = d["text"].lower()
        score = sum(text.count(term) for term in q)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for sc, d in scored[:top_k] if sc > 0]

def retrieve_transactions_context(query: str, txns: List[Transaction], top_k: int = 12) -> List[Dict[str, str]]:
    # 1) Prefer FAISS
    if has_faiss_index("tx_faiss") and os.getenv("OPENAI_API_KEY"):
        try:
            return semantic_search_faiss(query, top_k=top_k, name="tx_faiss")
        except Exception:
            pass
    # 2) Fallback to NPZ if present
    if has_index("tx_index") and os.getenv("OPENAI_API_KEY"):
        try:
            return semantic_search(query, top_k=top_k, filename="tx_index")
        except Exception:
            pass
    # 3) Keyword fallback
    docs = [{"id": t.id, "text": _pack_text(t)} for t in txns]
    return _keyword_rank(query, docs, top_k) or docs[:top_k]
