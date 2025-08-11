import os, json
from typing import Dict, Any
from .io import load_transactions
from .retrieval import retrieve_transactions_context
from .nlp_utils import parse_month, month_key, parse_last_n_months

def tool_sum_interest_month(transactions_path: str, month_text: str):
    tx = load_transactions(transactions_path)
    yr, mo = parse_month(month_text); ym = f"{yr:04d}-{mo:02d}" if (yr and mo) else None
    total = 0.0; ids = []
    for t in tx:
        if (t.transaction_type or '').upper() == 'INTEREST' and (ym is None or month_key(t.transaction_date_time) == ym):
            total += (t.amount or 0.0); ids.append(t.id)
    return {"total": round(total,2), "sources": ids[:25], "month": ym or "ALL"}

def tool_count_purchases_over(transactions_path: str, threshold: float, month_text: str | None = None):
    tx = load_transactions(transactions_path)
    ym = None
    if month_text:
        yr, mo = parse_month(month_text); 
        if yr and mo: ym = f"{yr:04d}-{mo:02d}"
    count = 0; ids = []
    for t in tx:
        if (t.transaction_type or '').upper() != 'PURCHASE': continue
        if abs(t.amount or 0.0) <= threshold: continue
        if ym and month_key(t.transaction_date_time) != ym: continue
        ids.append(t.id); count += 1
    return {"count": count, "sources": ids[:25], "month": ym or "ALL", "threshold": threshold}

def tool_rag_search(transactions_path: str, query: str, top_k: int = 12):
    tx = load_transactions(transactions_path)
    docs = retrieve_transactions_context(query, tx, top_k=top_k)
    return {"results": [{"id": d.get("id"), "text": d.get("text"), "score": float(d.get("score", 0.0))} for d in docs]}

def tool_interest_last_n_months(transactions_path: str, text: str):
    from .engine import _sum_interest_last_n_months
    tx = load_transactions(transactions_path)
    n = parse_last_n_months(text) or 6
    total, per_acct, ids = _sum_interest_last_n_months(tx, n)
    return {"total": total, "per_account": per_acct, "months": n, "sources": ids[:25]}

def tool_statement_last_n_months(transactions_path: str, text: str):
    from .engine import _statement_summary_last_n_months
    tx = load_transactions(transactions_path)
    n = parse_last_n_months(text) or 6
    stmt, ids = _statement_summary_last_n_months(tx, n)
    return {"months": n, "statement": stmt, "sources": ids[:25]}

def tool_get_account_balance(account_id: str):
    return {"accountId": account_id, "balance": 1234.56, "currency": "USD", "source": "stub"}

def tool_pay_bill(payee: str, amount: float, date: str | None = None, account_id: str | None = None):
    return {"status": "scheduled", "payee": payee, "amount": amount, "date": date or "next_business_day", "accountId": account_id or "default", "source": "stub"}

def _ensure_llamaindex():
    try:
        import llama_index  # noqa: F401
    except Exception as e:
        raise RuntimeError("llama-index is required. Install with `pip install llama-index`") from e

def build_agent(transactions_path: str = "data/transactions.json"):
    _ensure_llamaindex()
    from llama_index.core.tools import FunctionTool
    from llama_index.agent.openai import OpenAIAgent
    tools = [
        FunctionTool.from_defaults(fn=lambda q: tool_rag_search(transactions_path, q), name="rag_search"),
        FunctionTool.from_defaults(fn=lambda m: tool_sum_interest_month(transactions_path, m), name="sum_interest"),
        FunctionTool.from_defaults(fn=lambda thr, month=None: tool_count_purchases_over(transactions_path, thr, month), name="count_purchases_over"),
        FunctionTool.from_defaults(fn=lambda text: tool_interest_last_n_months(transactions_path, text), name="interest_last_n_months"),
        FunctionTool.from_defaults(fn=lambda text: tool_statement_last_n_months(transactions_path, text), name="statement_last_n_months"),
        FunctionTool.from_defaults(fn=tool_get_account_balance, name="get_account_balance"),
        FunctionTool.from_defaults(fn=tool_pay_bill, name="pay_bill"),
    ]
    system_prompt = ("You are a banking copilot focused on TRANSACTIONS. Use tools when helpful. "
                     "Prefer RAG search over transactions for evidence, and deterministic tools for math. "
                     "Return JSON with keys: answer, reasoning, sources.")
    agent = OpenAIAgent.from_tools(tools=tools, system_prompt=system_prompt, verbose=False)
    return agent

def ask_agent(query: str, transactions_path: str = "data/transactions.json") -> Dict[str, Any]:
    agent = build_agent(transactions_path)
    resp = agent.chat(query)
    try:
        content = str(resp.response)
        data = json.loads(content)
        if isinstance(data, dict) and all(k in data for k in ("answer","reasoning","sources")):
            return data
        return {"answer": content, "reasoning": "Agent response", "sources": []}
    except Exception:
        return {"answer": str(resp.response), "reasoning": "Agent response (non-JSON)", "sources": []}
