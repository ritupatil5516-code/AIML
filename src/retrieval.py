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

def retrieve_transactions_context(query: str, txns: List[Transaction], top_k: int = 12) -> List[Dict[str, str]]:
    # init
    docs: List[Dict[str, str]] = []

    # 1) FAISS semantic search
    if has_faiss_index("tx_faiss") and os.getenv("OPENAI_API_KEY"):
        try:
            docs.extend(semantic_search_faiss(query, top_k=top_k, name="tx_faiss"))
        except Exception:
            pass

    q = query.lower()

    # 2) Balance queries: pin the single latest POSTED txn (or latest in month)
    if "balance" in q or "ending balance" in q or "current balance" in q:
        yr, mo = parse_month(q)
        ym = f"{yr:04d}-{mo:02d}" if (yr and mo) else None

        pool = [
            t for t in txns
            if (t.transaction_status or "").upper() == "POSTED"
            and (ym is None or month_key(t.transaction_date_time) == ym)
        ] or [
            t for t in txns
            if (ym is None or month_key(t.transaction_date_time) == ym)
        ]

        if pool:
            latest = max(pool, key=lambda x: _dt_key(x.transaction_date_time))
            docs.append({"id": latest.id, "text": _pack_text(latest), "score": 1e12})  # pin to top

    # 3) “after my last payment …”: pin latest POSTED PAYMENT txn
    if "payment" in q:
        pays = [
            t for t in txns
            if (t.transaction_type or "").upper() == "PAYMENT"
            and (t.transaction_status or "").upper() == "POSTED"
        ]
        if pays:
            latest_pay = max(pays, key=lambda x: _dt_key(x.transaction_date_time))
            docs.append({"id": latest_pay.id, "text": _pack_text(latest_pay), "score": 1e13})  # even higher

    # 4) NPZ fallback only if still empty
    if not docs and has_index("tx_index") and os.getenv("OPENAI_API_KEY"):
        try:
            docs.extend(semantic_search(query, top_k=top_k, filename="tx_index"))
        except Exception:
            pass

    # 5) Keyword fallback only if still empty
    if not docs:
        base = [{"id": t.id, "text": _pack_text(t)} for t in txns]
        docs = _keyword_rank(query, base, top_k) or base[:top_k]

    # 6) De‑dup by id, sort by score desc, trim to top_k
    seen = set()
    out: List[Dict[str, str]] = []
    for d in sorted(docs, key=lambda x: x.get("score", 0.0), reverse=True):
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        out.append(d)
        if len(out) >= top_k:
            break
    return out