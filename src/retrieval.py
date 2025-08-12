import os
from typing import List, Dict
from .models import Transaction
from .nlp_utils import parse_month, month_key
from .semantic_index import has_index, semantic_search
from .faiss_index import has_faiss_index, semantic_search_faiss

def _pack_text(t: Transaction) -> str:
    return (
        f"[{t.id}] "
        f"type={t.transaction_type} | "
        f"status={t.transaction_status} | "
        f"date={t.transaction_date_time} | "
        f"amount={t.amount} {t.currency_code or ''} | "
        f"endingBalance={getattr(t, 'ending_balance', None)} | "
        f"merchant={t.merchant_name or ''} | accountId={t.account_id}"
    )

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
    # init
    docs: List[Dict[str, str]] = []

    # 1) Try FAISS semantic search
    if has_faiss_index("tx_faiss") and os.getenv("OPENAI_API_KEY"):
        try:
            docs.extend(semantic_search_faiss(query, top_k=top_k, name="tx_faiss"))
        except Exception:
            pass

    # 2) Balance-specific augmentation: ensure latest POSTED row(s) are present
    q = query.lower()
    if "balance" in q or "ending balance" in q or "current balance" in q:
        yr, mo = parse_month(q)
        ym = f"{yr:04d}-{mo:02d}" if (yr and mo) else None

        pool = [
            t for t in txns
            if (t.transaction_status or "").upper() == "POSTED"
               and (ym is None or month_key(t.transaction_date_time) == ym)
        ]
        # If no POSTED in that period, relax status
        if not pool:
            pool = [
                t for t in txns
                if (ym is None or month_key(t.transaction_date_time) == ym)
            ]

        pool.sort(key=lambda x: (x.transaction_date_time or ""))
        tail = pool[-3:]  # most recent few
        for t in tail:
            docs.append({"id": t.id, "text": _pack_text(t), "score": 1e9})

    # 3) Fallback to NPZ semantic search only if still empty
    if not docs and has_index("tx_index") and os.getenv("OPENAI_API_KEY"):
        try:
            docs.extend(semantic_search(query, top_k=top_k, filename="tx_index"))
        except Exception:
            pass

    # 4) Keyword fallback only if still empty
    if not docs:
        base = [{"id": t.id, "text": _pack_text(t)} for t in txns]
        docs = _keyword_rank(query, base, top_k) or base[:top_k]

    # 5) De-duplicate by id and keep highest scores first (respect top_k)
    seen = set()
    dedup: List[Dict[str, str]] = []
    for d in sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True):
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        dedup.append(d)
        if len(dedup) >= top_k:
            break

    return dedup
