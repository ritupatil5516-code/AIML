from typing import List, Dict
import os
from src.domain import load_glossary
import json  # add if missing

# src/prompts.py
import json
import os

def load_domain_glossary():
    glossary_path = os.getenv("DOMAIN_GLOSSARY_PATH", "glossary.json")
    if os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

DOMAIN_GLOSSARY = load_domain_glossary()

LATEST_TX_POLICY = """
Latest Transaction policy:
- “latest”, “most recent”, or “last transaction” means the row with the MAX `transactionDateTime` (ISO).
- Prefer POSTED over PENDING unless the user explicitly says to include pending.
- For “after my last payment”, filter to transactionType == "PAYMENT", then choose the MAX date.
- When answering, describe the chosen row: type, date/time, amount (with currency), merchantName (if any), and endingBalance.
- Return STRICT JSON: {"answer": string, "reasoning": string, "sources": [transactionId]}.
"""

BALANCE_RULES = """ 
When asked about 'current balance', 'ending balance', or 'balance in a specific month':
1. Always use the `endingBalance` field from the most recent POSTED transaction.
2. Ignore transactions with status 'PENDING'.
3. If a month/year is given, use the most recent POSTED transaction in that month.
4. Do not sum amounts to compute balance — balance is directly given in `endingBalance`.
5. If no POSTED transaction exists for that period, respond with: "Information not available in the provided data."
"""

FEWSHOTS = """
Example A
Q: What is my latest transaction?
Candidates: [{"transactionId":"t1","transactionDateTime":"2025-07-01T12:00:00Z"}, {"transactionId":"t2","transactionDateTime":"2025-08-03T09:00:00Z"}]
Answer:
{"answer":"(describe t2)","reasoning":"Picked max transactionDateTime = 2025-08-03T09:00:00Z","sources":["t2"]}

Example B
Q: After my last payment, what was the balance?
Candidates: [{"transactionId":"p1","transactionType":"PAYMENT","transactionStatus":"POSTED","transactionDateTime":"2025-08-10T10:00:00Z","endingBalance":501.64}, {"transactionId":"x2","transactionType":"PURCHASE","transactionStatus":"POSTED","transactionDateTime":"2025-08-11T18:00:00Z"}]
Answer:
{"answer":"501.64","reasoning":"Filtered to PAYMENT, picked latest by date (p1) and used its endingBalance","sources":["p1"]}
"""

SYSTEM_PROMPT = f"""
You are a banking assistant. Use ONLY provided candidate transactions and these rules.
{LATEST_TX_POLICY}
"""

def build_system_prompt():
    glossary_text = ""
    if DOMAIN_GLOSSARY:
        glossary_text = "Domain Glossary:\n" + json.dumps(DOMAIN_GLOSSARY, indent=2) + "\n\n"
    return f"""
    {BALANCE_RULES}\nY
You are a banking assistant specialized in TRANSACTIONS ONLY.
Rules:
1. Use ONLY the provided transaction context or tool results.
2. Prefer calling tools for math/filters; do not guess.
3. If info is missing, answer exactly: "Information not available in the provided data."
4. Respond in STRICT JSON with keys: answer (string), reasoning (string), sources (string[] of transaction IDs used).
{glossary_text}
{LATEST_TX_POLICY}
"""

SYSTEM_PROMPT = build_system_prompt()


def render_user_prompt_latest(query: str, latest_tx: dict) -> str:
    return (
        "Question: " + query + "\n"
        "LatestTransaction:\n" + json.dumps(latest_tx, ensure_ascii=False) + "\n\n"
        "Follow the Latest Transaction policy. "
        "Write a one‑sentence natural‑language answer describing the transaction "
        "(type, date/time, amount+currency, merchant if present, endingBalance). "
        "Then return STRICT JSON with keys: answer, reasoning, sources (use the transactionId)."
    )
