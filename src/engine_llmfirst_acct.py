# src/engine_llmfirst_acct.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any
from openai import OpenAI

from .models import Transaction, AccountSummary
from .retrieval_llmfirst import retrieve_candidates, pack_jsonl
from .retrieval_accounts import retrieve_accounts, pack_accounts_jsonl, looks_like_account_query
from .prompts_llmfirst import SYSTEM_LLM_FIRST_ACCOUNTS, render_llm_first_user_accounts

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm_first_accounts(query: str,
                           transactions: List[Transaction],
                           accounts: List[AccountSummary],
                           chat_history: List[Dict[str,str]] | None = None) -> Dict[str,Any]:

    # Retrieve candidates
    tx_cands = retrieve_candidates(query, transactions, top_k=120)
    acct_cands = retrieve_accounts(query, accounts, top_k=12) if looks_like_account_query(query) else accounts[:12]

    tx_jsonl   = pack_jsonl(tx_cands)
    acct_jsonl = pack_accounts_jsonl(acct_cands)

    messages = [{"role":"system","content":SYSTEM_LLM_FIRST_ACCOUNTS}]
    if chat_history:
        messages.extend([m for m in chat_history if m.get("role") in ("user","assistant")][-6:])
    messages.append({"role":"user","content":render_llm_first_user_accounts(query, tx_jsonl, acct_jsonl)})

    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        messages=messages,
        response_format={"type":"json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content
    try:
        js = json.loads(raw)
    except Exception:
        return {"answer": raw, "reasoning": "LLM returned non-JSON", "sources": []}

    # Final response uses the IDs the LLM picked; we don't “fix” account numbers,
    # but you could verify numeric fields by re-reading those account rows here.
    sources = (js.get("selected_tx_ids") or []) + (js.get("selected_account_ids") or [])
    return {
        "answer": js.get("answer") or "Information not available in the provided data.",
        "reasoning": js.get("reasoning") or "Reasoned over TX + Account summaries.",
        "sources": sources
    }