"""
Smoke tests for TX Copilot (LLM-first).
Usage:
  uv run python scripts/smoke_tests.py  # or: python scripts/smoke_tests.py
"""

from __future__ import annotations
import os, json, time
from typing import List, Dict, Any

# -------- project imports (adjust paths if needed) --------
from src.io import load_transactions, load_account_summaries
from src.engine_llmfirst_acct import ask_llm_first_accounts

DATA_DIR = os.getenv("DATA_DIR", "data")
TX_PATH = os.path.join(DATA_DIR, "transactions.json")
ACCT_PATH = os.path.join(DATA_DIR, "account-summary.json")

QUESTIONS: List[str] = [
    # Transactions
    "Total amount credited this year?",
    "How much did I spend in July 2025?",
    "Total purchase amount in Aug 2025",
    "Show the biggest transaction this year",
    "List all transactions above $500 in 2025",
    "When was my last payment and how much?",
    "Show transactions for Southwest in 2025",
    "How much did I spend on FUEL in 2023?",
    "Total credits in 2023-09",
    "Show all transactions between 2025-07-01 and 2025-07-31",

    # Accounts
    "What is my current balance?",
    "What is my available credit and credit limit for account ending 0269?",
    "What is my minimum due and payment due date?",
    "Which accounts are past due or overdue?",
    "Which accounts have the BLOCKED_SPEND flag?",
    "Show my latest account summary",

    # Mixed
    "For account ending 0269, list July 2025 transactions",
    "Across all accounts, how much have I spent in USD this year?",
    "List overdue accounts with total balance and payment due date",
    "What is my total credit limit across all accounts?",
    "After my last payment, what was the ending balance of the next posted transaction?",
]

def _short(s: str, n: int = 100) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"

def load_json_safe(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Ensure API key and model present
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running smoke tests.")
    os.environ.setdefault("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    # Load data
    tx = load_transactions(TX_PATH)
    accts = load_account_summaries(ACCT_PATH)
    print(f"Loaded: {len(tx)} transactions, {len(accts)} account summaries")

    # Run
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, q in enumerate(QUESTIONS, 1):
        q0 = time.time()
        res = ask_llm_first_accounts(q, tx, accts, chat_history=None)
        dt = time.time() - q0
        results.append({"q": q, "res": res, "ms": int(dt * 1000)})
        print(f"[{i:02d}] {q} -> {res.get('answer')} (ids={len(res.get('sources', []))}, {dt:.2f}s)")

    total = time.time() - t0
    print(f"\nCompleted {len(QUESTIONS)} questions in {total:.2f}s")

    # Simple quality counters
    empty_sources = sum(1 for r in results if not r["res"].get("sources"))
    long_answers = sum(1 for r in results if len((r["res"].get("answer") or "")) > 240)

    print(f"Empty sources: {empty_sources}/{len(results)}  |  Very long answers: {long_answers}/{len(results)}")

    # Optional: write a JSON report for CI artifacts
    out_dir = os.getenv("REPORT_DIR", "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()