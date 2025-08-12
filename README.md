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

## Transactions (spend, credits, payments)
	•	What is the total amount credited this year?
	•	How much did I spend in July 2025?
	•	Total purchase amount in Aug 2025.
	•	What’s my largest transaction this year?
	•	List all transactions over $500 in 2025.
	•	When was my last payment and how much?
	•	Show transactions for merchant “Southwest” in 2025.
	•	How much did I spend on FUEL in 2023?
	•	Sum of credits in 2023-09.
	•	Show all transactions between 2025-07-01 and 2025-07-31.

## Accounts (balances, due amounts, flags)
	•	What is my current balance?
	•	What is my available credit and credit limit for account ending 0269?
	•	What is my minimum due and payment due date?
	•	Which accounts are PAST_DUE or OVERDUE?
	•	Which accounts have the BLOCKED_SPEND flag?
	•	Show the latest account summary (newest lastUpdatedDate).

## Mixed (TX + Accounts)
	•	For account ending 0269, list all July 2025 transactions.
	•	Across all accounts, how much have I spent in USD this year?
	•	List overdue accounts with their totalBalance and paymentDueDate.
	•	What is my total credit limit across all accounts?
	•	After my last payment, what was the ending balance of the next posted transaction?

⸻
