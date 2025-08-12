from datetime import datetime
from typing import List, Dict, Any, Optional, Iterable
from .models import Transaction
from .domain import get_field_doc

def _to_dict(i):
    if i is None:
        return {}
    if hasattr(i, "model_dump"):        # Pydantic v2
        return i.model_dump(by_alias=True)
    if hasattr(i, "dict"):              # Pydantic v1
        return i.dict(by_alias=True)
    if isinstance(i, dict):
        return i
    if isinstance(i, str):
        try:
            return json.loads(i)
        except Exception:
            return {}
    return {}

def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str or not isinstance(dt_str, str):
        return None
    s = dt_str.strip()
    # tolerate trailing Z and fractional seconds
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(s.split(".")[0].replace("Z",""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

def _match_month_year(d: dict, month: Optional[str], year: Optional[str]) -> bool:
    """month = 'YYYY-MM' or None, year = 'YYYY' or None."""
    if not month and not year:
        return True
    dt = _parse_iso(d.get("transactionDateTime") or d.get("transaction_date_time"))
    if not dt:
        # last‑ditch substring fallback (handles well‑formed ISO strings)
        s = (d.get("transactionDateTime") or d.get("transaction_date_time") or "")
        if month and isinstance(s, str) and s.startswith(month):
            return True
        if year and isinstance(s, str) and s.startswith(year):
            return True
        return False
    if month:
        try:
            y, m = month.split("-")
            return dt.year == int(y) and dt.month == int(m)
        except Exception:
            return False
    if year:
        try:
            return dt.year == int(year)
        except Exception:
            return False
    return True

def _is_posted(d: dict) -> bool:
    return (d.get("transactionStatus") or d.get("transaction_status") or "").upper() == "POSTED"

def _is_credit(d: dict) -> bool:
    """Credit = debitCreditIndicator == -1 (string or int)."""
    ind = d.get("debitCreditIndicator")
    try:
        return int(ind) == -1
    except Exception:
        return False

def _is_debit(d: dict) -> bool:
    """Debit = debitCreditIndicator == 1 (string or int)."""
    ind = d.get("debitCreditIndicator")
    try:
        return int(ind) == 1
    except Exception:
        return False

# ---------- totals ----------
def sum_credits(transactions: Iterable, month: str | None = None, year: str | None = None) -> float:
    """Total of POSTED credits. Credit strictly = debitCreditIndicator == -1."""
    total = 0.0
    for t in transactions:
        d = _to_dict(t)
        if not _is_posted(d):
            continue
        if not _match_month_year(d, month, year):
            continue
        if not _is_credit(d):
            continue
        try:
            total += float(d.get("amount") or 0.0)
        except Exception:
            pass
    return float(total)

def sum_debits(transactions: Iterable, month: str | None = None, year: str | None = None) -> float:
    """Total of POSTED debits. Debit strictly = debitCreditIndicator == 1."""
    total = 0.0
    for t in transactions:
        d = _to_dict(t)
        if not _is_posted(d):
            continue
        if not _match_month_year(d, month, year):
            continue
        if not _is_debit(d):
            continue
        try:
            # amounts may be positive; we sum their absolute value
            amt = float(d.get("amount") or 0.0)
            total += abs(amt)
        except Exception:
            pass
    return float(total)

def sum_payments(transactions: Iterable, month: str | None = None, year: str | None = None) -> float:
    """Total of POSTED PAYMENT transactions (by type). Use when business asks 'total payment ...'."""
    total = 0.0
    for t in transactions:
        d = _to_dict(t)
        if not _is_posted(d):
            continue
        if not _match_month_year(d, month, year):
            continue
        if (d.get("transactionType") or d.get("transaction_type") or "").upper() != "PAYMENT":
            continue
        try:
            total += float(d.get("amount") or 0.0)
        except Exception:
            pass
    return float(total)

def explain_field(field_name: str) -> dict | None:
    doc = get_field_doc(field_name)
    if not doc:
        return None
    return {"field": field_name, "explanation": doc}

def filter_transactions(transactions: List[Transaction], 
                        min_amount: float | None = None, 
                        max_amount: float | None = None,
                        transaction_type: str | None = None,
                        merchant_name: str | None = None,
                        status: str | None = None) -> List[Dict[str, Any]]:
    res = []
    for t in transactions:
        if transaction_type and (t.transaction_type or "").upper() != transaction_type.upper(): continue
        if merchant_name and (t.merchant_name or "").lower() != merchant_name.lower(): continue
        if status and (t.transaction_status or "").upper() != status.upper(): continue
        if min_amount is not None and (t.amount or 0) < min_amount: continue
        if max_amount is not None and (t.amount or 0) > max_amount: continue
        res.append({"transactionId": t.id, "amount": t.amount, "type": t.transaction_type, "date": t.transaction_date_time})
    return res

def sum_amounts(items: List[Dict[str, Any]]) -> float:
    return float(sum((i.get("amount") or 0.0) for i in items))

def count_items(items: List[Dict[str, Any]]) -> int:
    return int(len(items))

def get_transaction_by_id(transactions: List[Transaction], txn_id: str) -> Dict[str, Any] | None:
    for t in transactions:
        if (t.id or "") == (txn_id or ""):
            return {"transactionId": t.id, "amount": t.amount, "type": t.transaction_type, "date": t.transaction_date_time, "status": t.transaction_status, "currency": t.currency_code, "merchant": t.merchant_name}
    return None
