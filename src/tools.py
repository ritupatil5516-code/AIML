from typing import List, Dict, Any
from .models import Transaction
from .domain import get_field_doc

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



import json
from typing import Iterable

def _to_dict(i):
    if i is None:
        return {}
    if hasattr(i, "model_dump"):
        return i.model_dump(by_alias=True)
    if hasattr(i, "dict"):
        return i.dict(by_alias=True)
    if isinstance(i, dict):
        return i
    if isinstance(i, str):
        try:
            return json.loads(i)
        except Exception:
            return {}
    return {}

def sum_amounts(items: Iterable) -> float:
    total = 0.0
    for i in items:
        d = _to_dict(i)
        try:
            total += float(d.get("amount") or 0.0)
        except Exception:
            pass
    return float(total)

def sum_credits(transactions: Iterable, month: str | None = None) -> float:
    """Sum all POSTED credits across dataset. Credit = debitCreditIndicator == -1 or amount > 0."""
    total = 0.0
    for t in transactions:
        d = _to_dict(t)
        status = (d.get("transactionStatus") or d.get("transaction_status") or "").upper()
        if status != "POSTED":
            continue
        ind = d.get("debitCreditIndicator")
        try:
            ind = int(ind)
        except Exception:
            ind = None
        amt = float(d.get("amount") or 0.0)
        is_credit = (ind == -1) or (amt > 0)
        if not is_credit:
            continue
        if month:
            dt = (d.get("transactionDateTime") or d.get("transaction_date_time") or "")
            if not (isinstance(dt, str) and dt[:7] == month):
                continue
        total += amt
    return float(total)
