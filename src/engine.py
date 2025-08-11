import os, json
from typing import Any, Dict, List, Tuple
from .io import load_transactions
from .retrieval import retrieve_transactions_context
from .prompts import SYSTEM_PROMPT, render_user_prompt
from . import tools as tx_tools
from .nlp_utils import parse_month, month_key, parse_last_n_months

USE_LLM_TOOLS = os.getenv('USE_LLM_TOOLS', 'true').lower() == 'true'

def _tool_schema():
    return [
      {"type":"function","function":{"name":"filter_transactions","description":"Filter transactions by amount/type/status.","parameters":{"type":"object","properties":{"min_amount":{"type":"number"},"max_amount":{"type":"number"},"transaction_type":{"type":"string"},"merchant_name":{"type":"string"},"status":{"type":"string"}},"additionalProperties": False}}},
      {"type":"function","function":{"name":"sum_amounts","description":"Sum the 'amount' field of items.","parameters":{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"transactionId":{"type":"string"},"amount":{"type":"number"}},"required":["transactionId","amount"],"additionalProperties": True}}},"required":["items"],"additionalProperties": False}}},
      {"type":"function","function":{"name":"count_items","description":"Count items in an array.","parameters":{"type":"object","properties":{"items":{"type":"array","items":{"type":"object"}}},"required":["items"],"additionalProperties": False}}},
      {"type":"function","function":{"name":"get_transaction_by_id","description":"Get one transaction by ID.","parameters":{"type":"object","properties":{"txn_id":{"type":"string"}},"required":["txn_id"],"additionalProperties": False}}},
      {"type":"function","function":{"name":"explain_field","description":"Explain what a company-specific field means.","parameters":{"type":"object","properties":{"field_name":{"type":"string"}},"required":["field_name"],"additionalProperties": False}}}
    ]

def _call_tool(name: str, args: Dict[str, Any], state: Dict[str, Any]):
    tx = state["transactions"]
    if name == "filter_transactions": return tx_tools.filter_transactions(tx, **args)
    if name == "sum_amounts": return tx_tools.sum_amounts(args.get("items", []))
    if name == "count_items": return tx_tools.count_items(args.get("items", []))
    if name == "get_transaction_by_id": return tx_tools.get_transaction_by_id(tx, args.get("txn_id"))
    if name == "explain_field": return tx_tools.explain_field(args.get("field_name"))
    raise ValueError(f"Unknown tool: {name}")

# Deterministic helpers
def _sum_interest(transactions, ym: str | None) -> Tuple[float, List[str]]:
    total, ids = 0.0, []
    for t in transactions:
        if (t.transaction_type or '').upper() == 'INTEREST' and (ym is None or month_key(t.transaction_date_time) == ym):
            total += (t.amount or 0.0); ids.append(t.id)
    return round(total,2), ids

def _count_purchases_over(transactions, threshold: float, ym: str | None) -> Tuple[int, List[str]]:
    ids = []
    for t in transactions:
        if (t.transaction_type or '').upper() != 'PURCHASE': continue
        if abs(t.amount or 0.0) <= threshold: continue
        if ym and month_key(t.transaction_date_time) != ym: continue
        ids.append(t.id)
    return len(ids), ids

def _most_recent_month(transactions) -> str | None:
    months = sorted({month_key(t.transaction_date_time) for t in transactions})
    return months[-1] if months else None

def _months_in_range(transactions, last_n: int):
    latest = None
    from datetime import datetime
    for t in transactions:
        try:
            d = datetime.fromisoformat((t.transaction_date_time or '').replace('Z','+00:00'))
            if latest is None or d > latest: latest = d
        except Exception: continue
    if latest is None:
        from datetime import datetime as _dt
        latest = _dt.utcnow()
    months = []
    y = latest.year; m = latest.month
    for _ in range(last_n):
        months.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0: m = 12; y -= 1
    return set(months)

def _sum_interest_last_n_months(transactions, last_n: int):
    months = _months_in_range(transactions, last_n)
    total = 0.0; ids = []; per_account = {}
    for t in transactions:
        if (t.transaction_type or '').upper() == 'INTEREST' and month_key(t.transaction_date_time) in months:
            total += (t.amount or 0.0); ids.append(t.id)
            acct = t.account_id or 'unknown'
            per_account.setdefault(acct, 0.0); per_account[acct] += (t.amount or 0.0)
    per_account = {k: round(v, 2) for k,v in per_account.items()}
    return round(total,2), per_account, ids

def _statement_summary_last_n_months(transactions, last_n: int):
    months = _months_in_range(transactions, last_n)
    summary = {}; ids = []
    for t in transactions:
        mk = month_key(t.transaction_date_time)
        if mk not in months: continue
        ids.append(t.id)
        d = summary.setdefault(mk, {"inflow":0.0,"outflow":0.0,"net":0.0,"count":0})
        amt = t.amount or 0.0
        if amt >= 0: d["inflow"] += amt
        else: d["outflow"] += -amt
        d["net"] += amt
        d["count"] += 1
    for mk, d in summary.items():
        d["inflow"] = round(d["inflow"],2); d["outflow"] = round(d["outflow"],2); d["net"] = round(d["net"],2)
    ordered = dict(sorted(summary.items(), key=lambda kv: kv[0], reverse=True))
    return ordered, ids

def _maybe_handle_deterministic(query: str, transactions):
    q = query.lower()
    # interest month total
    if 'interest' in q and ('total' in q or 'sum' in q or 'amount' in q):
        yr, mo = parse_month(q); ym = f"{yr:04d}-{mo:02d}" if (yr and mo) else None
        total, ids = _sum_interest(transactions, ym)
        return {"answer": f"{total}", "reasoning": "Summed INTEREST amounts" + (f" in {ym}" if ym else " across all months"), "sources": ids[:25]}
    # purchases over
    import re
    m = re.search(r'over\s*\$?\s*(\d+(?:\.\d+)?)', q)
    if 'purchase' in q and m:
        threshold = float(m.group(1))
        yr, mo = parse_month(q); ym = f"{yr:04d}-{mo:02d}" if (yr and mo) else None
        if ym is None and 'in a month' in q: ym = _most_recent_month(transactions)
        count, ids = _count_purchases_over(transactions, threshold, ym)
        when_txt = f" in {ym}" if ym else " across all months"
        return {"answer": f"{count}", "reasoning": f"Counted PURCHASE where |amount| > {threshold}{when_txt}.", "sources": ids[:25]}
    # last n months: interest & statement
    n = parse_last_n_months(q)
    if n and 'interest' in q:
        total, per_acct, ids = _sum_interest_last_n_months(transactions, n)
        return {"answer": json.dumps({"total_interest": total, "per_account": per_acct}), "reasoning": f"Summed INTEREST over last {n} months", "sources": ids[:25]}
    if n and ('statement' in q or 'summary' in q):
        stmt, ids = _statement_summary_last_n_months(transactions, n)
        return {"answer": json.dumps({"months": stmt}), "reasoning": f"Computed inflow/outflow/net for last {n} months", "sources": ids[:25]}
    return None

def ask_tx(query: str, use_llm: bool = True, transactions_path: str = "transactions.json", chat_history: list | None = None):
    transactions = load_transactions(transactions_path)
    det = _maybe_handle_deterministic(query, transactions)
    if det is not None: return det

    ctx = retrieve_transactions_context(query, transactions, top_k=12)
    if not use_llm:
        return {"answer":"LLM disabled","reasoning":"", "sources":[d["id"] for d in ctx]}

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    messages = [{"role":"system","content": SYSTEM_PROMPT}]
    if chat_history:
        for m in chat_history[-12:]:
            role = m.get("role")
            if role in ("user","assistant"):
                content = m.get("content")
                if not isinstance(content, str): content = json.dumps(content)
                messages.append({"role": role, "content": content})
    messages.append({"role":"user","content": render_user_prompt(query, ctx)})

    kwargs = {"model": os.getenv("CHAT_MODEL","meta-llama/Llama-3.3-70B-Instruct"), "messages": messages, "response_format":{"type":"json_object"}}
    if USE_LLM_TOOLS:
        kwargs["tools"] = _tool_schema()
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)

    msg = resp.choices[0].message
    if getattr(msg, "tool_calls", None):
        messages.append({"role":"assistant","content": msg.content or "", "tool_calls": msg.tool_calls})
        state = {"transactions": transactions}
        for tc in msg.tool_calls[:4]:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = _call_tool(name, args, state)
            messages.append({"role":"tool","tool_call_id": tc.id, "content": json.dumps(result)})
        resp = client.chat.completions.create(**kwargs | {"messages": messages})
        msg = resp.choices[0].message

    try:
        return json.loads(msg.content)
    except Exception:
        return {"answer": msg.content, "reasoning":"", "sources":[d["id"] for d in ctx]}
