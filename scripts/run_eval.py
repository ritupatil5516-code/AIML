#!/usr/bin/env python3
"""
Run a quick eval: ask the copilot each eval question and score exact match (or numeric tolerance).

Usage:
  python scripts/run_eval.py --data-dir data_synth --transactions data_synth/transactions.json
"""
import json, argparse, math
from pathlib import Path

from src.engine import ask_tx

def is_number(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def score(expected, got):
    if expected == got:
        return True
    # numeric close
    if is_number(expected) and is_number(got):
        return math.isclose(float(expected), float(got), rel_tol=1e-6, abs_tol=0.01)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transactions", required=True)
    args = ap.parse_args()

    eval_path = Path(args.data_dir) / "tx_qa_eval.jsonl"
    total = 0
    correct = 0

    with open(eval_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            exp = ex["answer"]
            res = ask_tx(q, transactions_path=args.transactions)
            got = res.get("answer", "")
            ok = score(exp, got)
            total += 1
            correct += 1 if ok else 0
            print(f"Q: {q}\n  expected: {exp}\n  got: {got}\n  {'OK' if ok else 'MISS'}\n")

    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")

if __name__ == "__main__":
    main()
