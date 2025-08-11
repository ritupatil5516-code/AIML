import json, os
from typing import List
from .models import Transaction

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_synth")
def load_transactions(path: str = "transactions.json") -> List[Transaction]:
    p = path if os.path.isabs(path) else os.path.join(DATA_DIR, path)
    if not os.path.exists(p): raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["transactions"] if isinstance(data, dict) and "transactions" in data else data
    return [Transaction(**t) for t in items]
