import os
from typing import List, Dict
from datetime import datetime
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


def _parse_dt(dt_str):
    if not dt_str:
        return datetime.min
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(dt_str.split(".")[0].replace("Z",""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.min

def retrieve_transactions_context(query: str, txns: list, top_k: int = 12):
    docs = []

    q_lower = query.lower()

    # === FAISS or existing semantic search ===
    try:
        if has_faiss_index("tx_faiss") and os.getenv("OPENAI_API_KEY"):
            docs.extend(semantic_search_faiss(query, top_k=top_k, name="tx_faiss"))
    except Exception:
        pass

    # === Pin latest POSTED payment if asking about payment ===
    if "payment" in q_lower:
        payments = [t for t in txns if (t.transactionType == "PAYMENT" and t.transactionStatus == "POSTED")]
        if payments:
            latest_payment = max(payments, key=lambda x: _parse_dt(t.transactionDateTime))
            docs.append({"id": latest_payment.transactionId, "text": _pack_text(latest_payment), "score": 1e12})

    # === Pin latest POSTED txn for ending balance queries ===
    if "ending balance" in q_lower or "current balance" in q_lower:
        posted = [t for t in txns if t.transactionStatus == "POSTED"]
        if posted:
            latest_txn = max(posted, key=lambda x: _parse_dt(t.transactionDateTime))
            docs.append({"id": latest_txn.transactionId, "text": _pack_text(latest_txn), "score": 1e11})

    return docs