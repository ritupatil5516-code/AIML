from typing import List, Dict
SYSTEM_PROMPT = """You are a banking assistant specialized in TRANSACTIONS ONLY.
Rules:
1. Use ONLY the provided transaction context or tool results.
2. Prefer calling tools for math/filters; do not guess.
3. If info is missing, answer exactly: "Information not available in the provided data."
4. Respond in STRICT JSON with keys: answer (string), reasoning (string), sources (string[] of transaction IDs used).
"""
def render_user_prompt(query: str, context_docs: List[Dict[str, str]]) -> str:
    ctx = "\n".join([f"[{d['id']}] {d['text']}" for d in context_docs])
    return f"""Question: {query}
Context (transactions only):
{ctx}
Reply in STRICT JSON with keys: answer, reasoning, sources."""
