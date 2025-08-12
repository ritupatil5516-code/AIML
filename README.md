# TX Copilot — Banking Transactions Assistant

Production-ready **banking copilot** with:
- **Semantic RAG** over transactions using **FAISS Flat** (exact cosine search)
- **LLaMA** for answers (chat model)
- **BGE** for embeddings (retrieval)
- **LlamaIndex Agent** (optional) for tool-driven flows
- **Company glossary** to explain domain fields & rules
- **Streamlit** chat UI with session history

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your_key"
export CHAT_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export EMBED_MODEL="BAAI/bge-en-icl"

python scripts/build_faiss_index.py --transactions data/transactions.json
streamlit run streamlit_app.py
```

# Transaction Query Examples

Below is a list of example questions you can ask the TX Copilot (Semantic RAG) to explore transaction data.

## General Transaction Queries
1. What is my current account balance?
2. How many transactions have I made this month?
3. Show me my last 5 transactions.
4. What is the highest amount I spent in a single transaction?
5. Which merchant did I spend the most money on this month?
6. What is my average transaction amount this year?
7. Show all transactions above $500.
8. How many transactions were made using my credit card?
9. What is the total amount spent this year?
10. Which merchants did I visit more than 5 times this year?

## Credit/Debit Specific Queries
11. What is the total credited amount this month?
12. What is the total debited amount this month?
13. Show me all credited transactions in the past 3 months.
14. How many debit transactions did I make this week?
15. Show me the largest credited transaction in 2024.

## Time-Based Queries
16. Show all transactions from August 2025.
17. What is the total spent in Q1 2024?
18. How much money did I receive in 2023?
19. Show transactions for the last 7 days.
20. What was my first transaction in 2022?

## Merchant-Specific Queries
21. How much did I spend at Amazon this year?
22. List all transactions from Starbucks in the last 6 months.
23. Which merchant category has the highest spend in 2025?
24. Show all fuel-related transactions this month.
25. Who are my top 5 merchants by spend amount?

## Pattern/Insight Queries
26. What percentage of my transactions are credits vs debits?
27. Show my monthly spending trend for 2024.
28. Which month had the highest total spend last year?
29. How many recurring transactions do I have?
30. What is the total amount of payments made toward my account this year?
