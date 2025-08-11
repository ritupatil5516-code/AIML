from typing import List, Dict
from .models import Transaction
def _txn_to_text(t: Transaction) -> str:
    return f"TRANSACTION id={t.id} | type={t.transaction_type} | status={t.transaction_status} | amount={t.amount} | date={t.transaction_date_time} | currency={t.currency_code} | merchant={t.merchant_name}"
def retrieve_transactions_context(query: str, txns: List[Transaction], top_k: int = 12) -> List[Dict[str, str]]:
    docs = [{"id": t.id, "text": _txn_to_text(t)} for t in txns]
    return docs[:top_k]
