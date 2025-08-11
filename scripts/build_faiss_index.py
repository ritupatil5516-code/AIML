#!/usr/bin/env python3
import argparse, os
from src.io import load_transactions
from src.faiss_index import build_faiss_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default="data/transactions.json")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL","BAAI/bge-en-icl"))
    ap.add_argument("--name", default="tx_faiss")
    args = ap.parse_args()

    tx = load_transactions(args.transactions)
    idx_path, meta_path = build_faiss_index(tx, embed_model=args.embed_model, name=args.name)
    print(f"Built FAISS index -> {idx_path}\nMeta -> {meta_path}")
