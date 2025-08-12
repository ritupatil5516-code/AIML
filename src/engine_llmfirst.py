from __future__ import annotations
import os, json
from typing import List, Dict, Any
from .retrieval_llmfirst import retrieve_candidates, pack_jsonl
from .prompts_llmfirst import SYSTEM_LLM_FIRST, render_llm_first_user
from .models import Transaction

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def verify_sum_from_ids(selected_ids: List[str], all_txns: List[Transaction]) -> float:
    id2t = {t.id: t for t in all_txns}
    total = 0.0
    for i in selected_ids:
        t = id2t.get(i)
        if not t:
            continue
        try:
            total += float(getattr(t, "amount", 0.0))
        except Exception:
            pass
    return round(total, 2)

def ask_llm_first(query: str, transactions: List[Transaction], chat_history: List[Dict[str,str]]|None=None) -> Dict[str,Any]:
    # 1) retrieve candidates (LLM will decide which ones to use)
    cands = retrieve_candidates(query, transactions, top_k=120)
    jsonl_rows = pack_jsonl(cands)

    # 2) build messages
    messages = [{"role": "system", "content": SYSTEM_LLM_FIRST}]
    if chat_history:
        messages.extend([m for m in chat_history if m.get("role") in ("user","assistant")][-6:])
    messages.append({"role": "user", "content": render_llm_first_user(query, jsonl_rows)})

    # 3) ask the LLM to select rows + compute
    resp = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content

    try:
        js = json.loads(raw)
    except Exception:
        return {"answer": raw, "reasoning": "LLM returned non-JSON", "sources": []}

    selected = js.get("selected_ids") or []
    # 4) verify math using the selected ids only
    verified_total = verify_sum_from_ids(selected, transactions)

    # 5) Final answer: keep their sentence, but replace number if needed
    ans = js.get("answer") or ""
    sum_guess = js.get("sum_guess")
    if isinstance(sum_guess, (int,float)) and abs(verified_total - float(sum_guess)) > 0.01:
        # adjust the sentence if it clearly contains the wrong number
        # (keep it simple—we don’t regex; we append a correction line)
        ans = f"{ans} (Verified total: {verified_total:.2f})"

    return {
        "answer": ans if ans else f"{verified_total:.2f}",
        "reasoning": js.get("reasoning") or "Selected rows from provided context and verified sum.",
        "sources": selected,
    }