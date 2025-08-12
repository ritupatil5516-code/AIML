from typing import List, Dict
import os
from src.domain import load_glossary

CREDIT_DEBIT_POLICY = """
Totals policy:
 CREDIT total → call `sum_credits`. Do not manually add amounts.
- DEBIT total → call `sum_debits`. Do not manually add amounts.
- PAYMENT total (by type) → call `sum_payments`. Do not manually add amounts.
- For “this month / last month / this year / last year”, do not guess dates; rely on runtime to infer month/year.
- For “total credited / total deposits / sum of credits”, call tool `sum_credits` (optionally pass month='YYYY-MM').
- For “total debited / total spends / sum of debits”, call tool `sum_debits` (optionally pass month='YYYY-MM').
- Do NOT add amounts manually; rely on tools for totals. Return JSON {answer, reasoning, sources}.
"""


BALANCE_RULES = """
Balance policy:
When asked about 'current balance', 'ending balance', 'account balance' or 'balance in a specific month':
1. Always use the `endingBalance` field from the most recent POSTED transaction.
2. Ignore transactions with status 'PENDING'.
3. If a month/year is given, use the most recent POSTED transaction in that month.
4. Do not sum amounts to compute balance — balance is directly given in `endingBalance`.
5. If no POSTED transaction exists for that period, respond with: "Information not available in the provided data."
"""

FEW_SHOTS = """
Example 1
Q: What is my current balance?
A: {"answer": "1543.22", "reasoning": "Used endingBalance from latest POSTED transaction", "sources": ["t-100"]}

Example 2
Q: Ending balance for Aug 2025?
A: {"answer": "980.50", "reasoning": "Used endingBalance from latest POSTED in 2025-08", "sources": ["t-200"]}

Example 3
Q: What is my balance now?
A: {"answer": "1192.45", "reasoning": "Used endingBalance from latest POSTED transaction; ignored pending", "sources": ["t-301"]}
"""


_g = load_glossary()
_use_gloss = os.getenv("USE_GLOSSARY_IN_PROMPT","true").lower()=="true"
_g_text = ""
if _use_gloss:
    fields = _g.get("fields", {})
    lines = [f"- {k}: {v.get('description','')}" for k,v in fields.items()]
    rules = _g.get("business_rules", {})
    rule_lines = [f"- {k}: {v}" for k,v in rules.items()]
    _g_text = "\n".join(["Company Domain Glossary:", *lines, "Business Rules:", *rule_lines])

SYSTEM_PROMPT = f"""{CREDIT_DEBIT_POLICY}\n{BALANCE_RULES}\nYou are a banking assistant specialized in TRANSACTIONS ONLY.
Rules:
1. Use ONLY the provided transaction context or tool results.
2. Prefer calling tools for math/filters; do not guess.
3. If info is missing, answer exactly: "Information not available in the provided data."
4. Respond in STRICT JSON with keys: answer (string), reasoning (string), sources (string[] of transaction IDs used).

{{_g_text}}
"""

def render_user_prompt(query: str, context_docs: List[Dict[str, str]]) -> str:
    ctx = "\n".join([f"[{d['id']}] {d['text']}" for d in context_docs])
    return f"""Question: {query}
Context (transactions only):
{ctx}
Reply in STRICT JSON with keys: answer, reasoning, sources."""
