from typing import List, Dict
import os
from .domain import load_glossary

_g = load_glossary()
_use_gloss = os.getenv("USE_GLOSSARY_IN_PROMPT","true").lower()=="true"
_g_text = ""
if _use_gloss:
    # keep prompt small: only field names & 1-liners
    fields = _g.get("fields", {})
    lines = [f"- {k}: {v.get("description","")}" for k,v in fields.items()]
    rules = _g.get("business_rules", {})
    rule_lines = [f"- {k}: {v}" for k,v in rules.items()]
    _g_text = "\n".join(["Company Domain Glossary:", *lines, "Business Rules:", *rule_lines])

SYSTEM_PROMPT = f"""You are a banking assistant specialized in TRANSACTIONS ONLY.
Rules:
1. Use ONLY the provided transaction context or tool results.
2. Prefer calling tools for math/filters; do not guess.
3. If info is missing, answer exactly: "Information not available in the provided data."
4. Respond in STRICT JSON with keys: answer (string), reasoning (string), sources (string[] of transaction IDs used).

{_g_text}
"""
def render_user_prompt(query: str, context_docs: List[Dict[str, str]]) -> str:
    ctx = "\n".join([f"[{d['id']}] {d['text']}" for d in context_docs])
    return f"""Question: {query}
Context (transactions only):
{ctx}
Reply in STRICT JSON with keys: answer, reasoning, sources."""
