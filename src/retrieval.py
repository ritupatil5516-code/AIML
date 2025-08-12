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

def _pack_json(t: Transaction) -> Dict[str, object]:
    return {
        "transactionId": t.id,
        "transactionType": t.transaction_type,
        "transactionStatus": t.transaction_status,
        "transactionDateTime": t.transaction_date_time,
        "amount": t.amount,
        "endingBalance": getattr(t, "ending_balance", None),
        "currencyCode": t.currency_code,
        "merchantName": t.merchant_name,
        "accountId": t.account_id,
    }

def _keyword_rank(query: str, docs: List[Dict[str, str]], top_k: int = 12) -> List[Dict[str, str]]:
    q = query.lower().split()
    scored = []
    for d in docs:
        text = d["text"].lower()
        score = sum(text.count(term) for term in q)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for sc, d in scored[:top_k] if sc > 0]

def _dt_key(iso: str | None) -> datetime:
    if not iso:
        return datetime.min
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(iso.split(".")[0].replace("Z",""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return datetime.min

def retrieve_candidate_txns(query: str, txns: List[Transaction], top_k: int = 30) -> List[Dict[str, object]]:
    """
    LLM-driven RAG: return structured candidate transactions that the LLM can reason over.
    - Try FAISS to get semantically similar ids.
    - Map back to Transaction objects and pack to JSON dicts.
    - Also include a small tail of the chronologically most recent POSTED transactions
      (just to guarantee recency is visible; we do NOT compute answers here).
    """
    by_id = {t.id: t for t in txns}
    candidates: List[Dict[str, object]] = []

    # 1) FAISS semantic hits → map to txns
    if has_faiss_index("tx_faiss") and os.getenv("OPENAI_API_KEY"):
        try:
            hits = semantic_search_faiss(query, top_k=top_k, name="tx_faiss")
            for h in hits:
                t = by_id.get(h.get("id"))
                if t:
                    candidates.append(_pack_json(t))
        except Exception:
            pass

    # 2) Append last ~10 POSTED transactions (LLM decides what “latest” means)
    posted = [t for t in txns if (t.transaction_status or "").upper() == "POSTED"]
    posted.sort(key=lambda x: (x.transaction_date_time or ""))
    for t in posted[-10:]:
        candidates.append(_pack_json(t))

    # 3) De-dup by transactionId, keep order of first appearance
    seen = set()
    uniq: List[Dict[str, object]] = []
    for c in candidates:
        tid = c.get("transactionId")
        if tid in seen:
            continue
        seen.add(tid)
        uniq.append(c)

    # cap to top_k (LLM doesn’t need everything)
    return uniq[:top_k]