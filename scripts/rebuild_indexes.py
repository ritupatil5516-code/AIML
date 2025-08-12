# scripts/rebuild_indexes.py
import json

from src.faiss_index_tx_acct import build_tx_index, build_account_index

TX_PATH = "data/transactions.json"
ACCT_PATH = "data/account-summary.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    tx_data = load_json(TX_PATH)
    tx_rows = tx_data.get("transactions", tx_data)
    build_tx_index(tx_rows, name="tx_faiss")

    acct_data = load_json(ACCT_PATH)
    acct_rows = acct_data.get("accounts", acct_data)
    build_account_index(acct_rows, name="acct_faiss")

    print("Indexes rebuilt: tx_faiss, acct_faiss")