from typing import List, Dict, Any
from .models import Transaction
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


from .domain import get_field_doc

def explain_field(field_name: str) -> dict | None:
    doc = get_field_doc(field_name)
    if not doc:
        return None
    return {"field": field_name, "explanation": doc}
