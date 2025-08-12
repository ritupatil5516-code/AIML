import json, os
from typing import List
from .models import Transaction, AccountSummary

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
def load_transactions(path: str = "transactions.json") -> List[Transaction]:
    p = path if os.path.isabs(path) else os.path.join(DATA_DIR, path)
    if not os.path.exists(p): raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["transactions"] if isinstance(data, dict) and "transactions" in data else data
    return [Transaction(**t) for t in items]

def load_account_summaries(path: str) -> list[AccountSummary]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # file can be {accounts:[...]} or a list
    items = data.get("accounts", data)
    out = []
    for it in items:
        try:
            out.append(AccountSummary(**it))
        except Exception:
            # be permissive on schema drift
            out.append(AccountSummary.model_validate(it))
    return out