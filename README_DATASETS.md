
# Dataset generation & eval for Transactions Copilot

## Generate
```bash
python scripts/generate_tx_dataset.py --out-dir data_synth --months 3 --accounts 2 --avg-per-month 80 --seed 42
```

## Evaluate
```bash
export OPENAI_API_KEY=...   # required
python scripts/run_eval.py --data-dir data_synth --transactions data_synth/transactions.json
```
