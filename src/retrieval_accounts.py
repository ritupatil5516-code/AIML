# src/retrieval_accounts.py
from __future__ import annotations
from typing import List, Dict, Any
from datetime import datetime

from .models import AccountSummary
from .faiss_index import has_faiss_index, semantic_search_faiss

ACCOUNT_HINTS = (
    "balance", "current balance", "total balance",
    "credit limit", "available credit", "limit",
    "past due", "minimum due", "payment due", "due date",
    "status", "flags", "overdue", "blocked", "installment",
    "billing cycle", "statement", "opened", "closed"
)

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

def looks_like_account_query(q: str) -> bool:
    q = (q or "").lower()
    return any(k in q for k in ACCOUNT_HINTS)

KEEP_FIELDS = [
    "accountId","accountNumberLast4","accountStatus","accountType","productType",
    "currentBalance","currentAdjustedBalance","totalBalance","availableCredit","creditLimit",
    "minimumDueAmount","pastDueAmount","highestPriorityStatus","balanceStatus",
    "paymentDueDate","paymentDueDateTime","billingCycleOpenDateTime","billingCycleCloseDateTime",
    "lastUpdatedDate","openedDate","closedDate","currencyCode","flags","subStatuses"
]

def to_row_dict(a: AccountSummary) -> Dict[str, Any]:
    if hasattr(a, "model_dump"):
        d = a.model_dump(by_alias=True)
    else:
        d = a.__dict__
    return {k: d.get(k) for k in KEEP_FIELDS}

def pack_accounts_jsonl(accts: List[AccountSummary]) -> str:
    import json
    return "\n".join(json.dumps(to_row_dict(a), separators=(",",":")) for a in accts)

def retrieve_accounts(query: str, accounts: List[AccountSummary], top_k=12) -> List[AccountSummary]:
    # If FAISS for accounts exists, use it; otherwise keyword + newest
    ids = []
    if has_faiss_index("acct_faiss"):
        try:
            ids = [d["id"] for d in semantic_search_faiss(query, top_k=top_k, name="acct_faiss")]
        except Exception:
            pass
    seen = set()
    pool = []
    # prioritize newest lastUpdatedDate
    accounts_sorted = sorted(accounts, key=lambda a: _dt_key(getattr(a, "lastUpdatedDate", None)), reverse=True)
    for a in accounts_sorted:
        if a.accountId in seen:
            continue
        seen.add(a.accountId or "")
        pool.append(a)
    # if FAISS provided ids, reorder to put them first
    idset = set(ids)
    pool.sort(key=lambda a: (a.accountId in idset, _dt_key(getattr(a,"lastUpdatedDate",None))), reverse=True)
    return pool[:top_k]