#!/usr/bin/env python3
import json, argparse, random, uuid, datetime as dt
from pathlib import Path

def iso(dtobj): return dtobj.strftime("%Y-%m-%dT%H:%M:%SZ")

def gen_transactions(n_accounts=2, months=3, avg_tx_per_month=80, seed=42):
    random.seed(seed)
    start = dt.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=90)
    transactions = []
    for a in range(n_accounts):
        account_id = f"acct-{a+1:03d}"
        for m in range(months):
            month_start = (start + dt.timedelta(days=30*m)).replace(day=1)
            days_in_month = 28 + (m % 4)
            count = max(10, int(random.gauss(avg_tx_per_month, max(1, avg_tx_per_month*0.15))))
            for _ in range(count):
                ttype = random.choice(["PURCHASE","DEPOSIT","WITHDRAWAL","INTEREST","FEE","REFUND","TRANSFER_IN","TRANSFER_OUT"])
                dtstamp = iso(month_start + dt.timedelta(days=random.randint(1, days_in_month)))
                amount = round(random.uniform(2, 3000), 2)
                if ttype in ("PURCHASE","WITHDRAWAL","TRANSFER_OUT","FEE"): amount = -abs(round(random.uniform(2, 300), 2))
                tx = {
                    "transactionId": "t-" + uuid.uuid4().hex[:10],
                    "accountId": account_id,
                    "transactionType": ttype,
                    "transactionStatus": random.choice(["POSTED","PENDING"]),
                    "amount": amount,
                    "transactionDateTime": dtstamp,
                    "currencyCode": "USD",
                }
                   
                if ttype in ("PURCHASE","REFUND"):
                    tx["merchantName"] = random.choice(["Amazon","Apple","Coffee Roasters","Shell","Whole Foods"])
                transactions.append(tx)
    transactions.sort(key=lambda x: x["transactionDateTime"])
    return transactions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--accounts", type=int, default=2)
    ap.add_argument("--avg-per-month", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build-index", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tx = gen_transactions(n_accounts=args.accounts, months=args.months, avg_tx_per_month=args.avg_per_month, seed=args.seed)
    (out / "transactions.json").write_text(json.dumps(tx, indent=2))

    if args.build_index:
        from src.io import load_transactions
        from src.faiss_index import build_faiss_index
        tx_m = load_transactions(str(out / "transactions.json"))
        build_faiss_index(tx_m)

    print(f"Wrote {len(tx)} transactions to {out}")

if __name__ == "__main__":
    main()
