#!/usr/bin/env python3
import argparse, os
from src.io import load_transactions
from src.semantic_index import build_index
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default="data/transactions.json")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL","text-embedding-3-small"))
    ap.add_argument("--name", default="tx_index")
    args = ap.parse_args()
    tx = load_transactions(args.transactions)
    path = build_index(tx, embed_model=args.embed_model, filename=args.name)
    print(f"Built index -> {path}")
