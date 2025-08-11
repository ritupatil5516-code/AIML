#!/usr/bin/env python3
"""
Generate synthetic transactions + QA (train/eval) for the Transactions Copilot.

Usage:
  python scripts/generate_tx_dataset.py --out-dir data_synth \
    --months 3 --accounts 2 --avg-per-month 80 --seed 42

Outputs:
  <out-dir>/transactions.json
  <out-dir>/tx_qa_train.jsonl
  <out-dir>/tx_qa_eval.jsonl
"""
import json, argparse, random, uuid, datetime as dt
from collections import defaultdict
from pathlib import Path

CURRENCY = "USD"
MERCHANTS = [
    ("Coffee Roasters", "Cafes"),
    ("Uber", "Transportation"),
    ("Spotify", "Entertainment"),
    ("Apple", "Electronics"),
    ("Amazon", "Online Retail"),
    ("Shell", "Gas"),
    ("Whole Foods", "Groceries"),
    ("PharmacyOne", "Pharmacy"),
    ("GymPlus", "Fitness"),
    ("City Utilities", "Utilities"),
]
CARD_TYPES = ["DEBIT", "CREDIT"]
STATUS = ["POSTED", "PENDING"]
TX_TYPES = ["PURCHASE","DEPOSIT","WITHDRAWAL","INTEREST","FEE","REFUND","TRANSFER_IN","TRANSFER_OUT"]

def iso(dtobj): return dtobj.strftime("%Y-%m-%dT%H:%M:%SZ")
def month_key(s): return s[:7]  # YYYY-MM

def gen_transactions(n_accounts=2, months=3, avg_tx_per_month=80, pct_installment=0.07, pct_recurring=0.12, seed=42):
    random.seed(seed)
    start = dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=90)
    transactions = []

    for a in range(n_accounts):
        account_id = f"acct-{a+1:03d}"
        last4 = f"{random.randint(1000,9999)}"
        card = random.choice(CARD_TYPES)

        for m in range(months):
            month_start = (start + dt.timedelta(days=30*m)).replace(day=1)
            days_in_month = 28 + (m % 4)
            count = max(10, int(random.gauss(avg_tx_per_month, max(1, avg_tx_per_month*0.15))))

            # some installment plans
            plan_ids = []
            for _ in range(max(0, int(count * pct_installment))):
                plan_id = "plan-" + uuid.uuid4().hex[:8]
                total_terms = random.choice([3, 6, 12])
                purchase_amt = round(random.uniform(150, 1500), 2)
                monthly = round(purchase_amt / total_terms, 2)
                plan_ids.append((plan_id, total_terms, purchase_amt, monthly))

                t = {
                    "transactionId": "t-" + uuid.uuid4().hex[:10],
                    "accountId": account_id,
                    "transactionType": "PURCHASE",
                    "transactionStatus": "POSTED",
                    "amount": -monthly,
                    "transactionDateTime": iso(month_start + dt.timedelta(days=random.randint(1,5))),
                    "currencyCode": CURRENCY,
                    "displayTransactionType": "Installment Charge",
                    "merchantName": random.choice(["Apple","Amazon","Best Buy","Furniture Hub"]),
                    "merchantCategoryName": "Electronics",
                    "cardType": card,
                    "lastFourDigits": last4,
                    "isRecurring": True,
                    "isInstallmentConversionEligible": False,
                    "installmentPlanId": plan_id,
                    "installmentTermNumber": 1,
                    "installmentTermTotal": total_terms,
                    "installmentPurchaseAmount": purchase_amt,
                    "installmentMonthlyAmount": monthly
                }
                transactions.append(t)

            # recurring subscriptions
            for merch, cat in random.sample(MERCHANTS, k=min(3, len(MERCHANTS))):
                if random.random() < pct_recurring:
                    amt = round(random.uniform(5, 25), 2) if "Coffee" in merch else round(random.uniform(5, 20), 2)
                    t = {
                        "transactionId": "t-" + uuid.uuid4().hex[:10],
                        "accountId": account_id,
                        "transactionType": "PURCHASE",
                        "transactionStatus": random.choice(STATUS),
                        "amount": -amt,
                        "transactionDateTime": iso(month_start + dt.timedelta(days=random.randint(1,3))),
                        "currencyCode": CURRENCY,
                        "displayTransactionType": "Subscription",
                        "merchantName": merch,
                        "merchantCategoryName": cat,
                        "cardType": card,
                        "lastFourDigits": last4,
                        "isRecurring": True
                    }
                    transactions.append(t)

            # general mix
            for _ in range(count):
                ttype = random.choices(TX_TYPES, weights=[40,8,8,3,2,3,3,3], k=1)[0]
                dtstamp = iso(month_start + dt.timedelta(days=random.randint(1, days_in_month)))
                merch, cat = random.choice(MERCHANTS)
                if ttype == "PURCHASE":
                    amount = round(-random.uniform(2, 250), 2)
                elif ttype in ("WITHDRAWAL","TRANSFER_OUT","FEE"):
                    amount = round(-random.uniform(5, 500), 2)
                else:
                    amount = round(random.uniform(2, 3000), 2)
                t = {
                    "transactionId": "t-" + uuid.uuid4().hex[:10],
                    "accountId": account_id,
                    "transactionType": ttype,
                    "transactionStatus": random.choice(STATUS),
                    "amount": amount,
                    "transactionDateTime": dtstamp,
                    "currencyCode": CURRENCY,
                    "merchantName": merch if ttype in ("PURCHASE","REFUND") else None,
                    "merchantCategoryName": cat if ttype in ("PURCHASE","REFUND") else None,
                    "cardType": card if ttype in ("PURCHASE","WITHDRAWAL") else None,
                    "lastFourDigits": last4 if ttype in ("PURCHASE","WITHDRAWAL") else None,
                    "isRecurring": (random.random() < pct_recurring) if ttype=="PURCHASE" else False,
                    "isInstallmentConversionEligible": (ttype=="PURCHASE")
                }
                transactions.append(t)

            # extend a couple of future installments
            for (plan_id, total_terms, purchase_amt, monthly) in plan_ids:
                for n in range(2, min(total_terms, 4)+1):
                    t = {
                        "transactionId": "t-" + uuid.uuid4().hex[:10],
                        "accountId": account_id,
                        "transactionType": "PURCHASE",
                        "transactionStatus": "POSTED",
                        "amount": -monthly,
                        "transactionDateTime": iso(month_start + dt.timedelta(days=7 + 30*(n-1))),
                        "currencyCode": CURRENCY,
                        "displayTransactionType": "Installment Charge",
                        "merchantName": "Installment Plan",
                        "merchantCategoryName": "Electronics",
                        "cardType": card,
                        "lastFourDigits": last4,
                        "isRecurring": True,
                        "isInstallmentConversionEligible": False,
                        "installmentPlanId": plan_id,
                        "installmentTermNumber": n,
                        "installmentTermTotal": total_terms,
                        "installmentPurchaseAmount": purchase_amt,
                        "installmentMonthlyAmount": monthly
                    }
                    transactions.append(t)

    transactions.sort(key=lambda x: x["transactionDateTime"])
    return transactions

def build_index(transactions):
    idx = {"by_month": defaultdict(list), "by_type": defaultdict(list), "by_plan": defaultdict(list)}
    for t in transactions:
        idx["by_month"][month_key(t["transactionDateTime"])].append(t)
        idx["by_type"][t["transactionType"]].append(t)
        if t.get("installmentPlanId"):
            idx["by_plan"][t["installmentPlanId"]].append(t)
    return idx

def qa_sets(transactions):
    idx = build_index(transactions)
    months = sorted(set(month_key(t["transactionDateTime"]) for t in transactions))
    qa = []

    def q_count_purchases_over(amount, month=None):
        subset = transactions if month is None else idx["by_month"].get(month, [])
        ids = [t["transactionId"] for t in subset if t["transactionType"]=="PURCHASE" and t["amount"] < -amount]
        return {"question": f"How many PURCHASE transactions over ${amount} in {month or 'all months'}?",
                "answer": str(len(ids)), "reasoning": f"type=PURCHASE & amount<-{amount}", "sources": ids[:25]}

    def q_total_interest(month=None):
        subset = transactions if month is None else idx["by_month"].get(month, [])
        ids = [t["transactionId"] for t in subset if t["transactionType"]=="INTEREST"]
        total = round(sum(t["amount"] for t in subset if t["transactionType"]=="INTEREST"), 2)
        return {"question": f"What is the total INTEREST amount in {month or 'all months'}?",
                "answer": f"{total}", "reasoning": "sum amounts where type=INTEREST", "sources": ids[:25]}

    def q_has_installments():
        plan_ids = list(idx["by_plan"].keys())
        return {"question": "Do I have any installment transactions? If yes, list the plan IDs.",
                "answer": "Yes" if plan_ids else "No",
                "reasoning": "presence of installmentPlanId",
                "sources": [idx["by_plan"][p][0]["transactionId"] for p in plan_ids][:25],
                "extra": {"plans": plan_ids}}

    def q_installment_summary(plan_id):
        terms = idx["by_plan"].get(plan_id, [])
        if not terms: return None
        terms_sorted = sorted(terms, key=lambda t: t["installmentTermNumber"])
        total_terms = terms_sorted[0]["installmentTermTotal"]
        paid_terms = len(terms_sorted)
        monthly = terms_sorted[0]["installmentMonthlyAmount"]
        remaining_terms = max(0, total_terms - paid_terms)
        remaining_amount = round(monthly * remaining_terms, 2)
        ids = [t["transactionId"] for t in terms_sorted]
        return {"question": f"For installment plan {plan_id}, how many terms remain and how much is remaining?",
                "answer": json.dumps({"remaining_terms": remaining_terms, "remaining_amount": remaining_amount}),
                "reasoning": f"total={total_terms}, paid={paid_terms}, monthly={monthly}", "sources": ids[:25]}

    def q_top_merchants(k=3, month=None):
        subset = transactions if month is None else idx["by_month"].get(month, [])
        spend = defaultdict(float); seen = defaultdict(list)
        for t in subset:
            if t["transactionType"]=="PURCHASE" and (t["amount"] or 0) < 0 and t.get("merchantName"):
                spend[t["merchantName"]] += -t["amount"]
                seen[t["merchantName"]].append(t["transactionId"])
        tops = sorted(spend.items(), key=lambda x:x[1], reverse=True)[:k]
        ids = [tid for m,_ in tops for tid in seen[m][:5]]
        return {"question": f"Who are my top {k} merchants by spend in {month or 'all months'}?",
                "answer": json.dumps([{m: round(v,2)} for m,v in tops]),
                "reasoning": "sum spend by merchant over PURCHASE debits",
                "sources": ids[:25]}

    def q_monthly_net_flow(month):
        subset = idx["by_month"].get(month, [])
        total = round(sum(t["amount"] for t in subset), 2)
        ids = [t["transactionId"] for t in subset][:25]
        return {"question": f"What is my net flow (sum of amounts) in {month}?",
                "answer": f"{total}", "reasoning": "sum amounts for month", "sources": ids}

    qa.append(q_count_purchases_over(50))
    qa.append(q_total_interest())
    qa.append(q_has_installments())
    if idx["by_plan"]:
        any_plan = next(iter(idx["by_plan"].keys()))
        qa.append(q_installment_summary(any_plan))
    qa.append(q_top_merchants(5))
    for m in months:
        qa.append(q_monthly_net_flow(m))
        qa.append(q_count_purchases_over(20, month=m))
        qa.append(q_total_interest(month=m))

    split = max(5, int(len(qa)*0.6))
    return qa[:split], qa[split:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=2)
    ap.add_argument("--avg-per-month", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tx = gen_transactions(n_accounts=args.accounts, months=args.months, avg_tx_per_month=args.avg_per_month, seed=args.seed)
    (out / "transactions.json").write_text(json.dumps(tx, indent=2))
    train, evals = qa_sets(tx)
    with (out / "tx_qa_train.jsonl").open("w") as f:
        for ex in train: f.write(json.dumps(ex) + "\n")
    with (out / "tx_qa_eval.jsonl").open("w") as f:
        for ex in evals: f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(tx)} transactions, {len(train)} train QAs, {len(evals)} eval QAs to {out}")

if __name__ == "__main__":
    main()
