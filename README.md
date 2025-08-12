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


Questions:
Credit totals
	1.	Total credited this month?
	2.	Total credited last month?
	3.	Total credited in 2023?
	4.	Total credited in 2024?
	5.	Total credited in September 2023?
	6.	Total credited in July 2023?

⸻

Debit totals
	7.	Total debited this month?
	8.	Total debited last month?
	9.	Total debited in 2023?
	10.	Total debited in 2024?
	11.	Total debited in August 2025?
	12.	Total debited in March 2024?

⸻

Payments (transactionType = PAYMENT)
	13.	Total payment this month?
	14.	Total payment last month?
	15.	Total payment in 2023?
	16.	Total payment in 2024?
	17.	Total payment in September 2023?
	18.	Total payment in July 2023?

⸻

Mixed timeframe styles
	19.	How much was credited this year?
	20.	How much was debited this year?
	21.	What was the total payment received last year?
	22.	Show me the total credited in the previous month.
	23.	Give me the total payments for 2023-09.
	24.	Give me the total debits for 2023-07.

Balance-related
	1.	What is my current account balance?
	2.	What was my balance at the end of last month?
	3.	How much did my balance change in August 2025?

⸻

Transaction lookup
	4.	Show me my most recent transaction.
	5.	What was my last payment made?
	6.	When did I last receive money?
	7.	Show the biggest transaction this year.
	8.	What was my smallest debit this month?
	9.	Who did I pay the most to in 2023?

⸻

Credits (money in)
	10.	How much have I received this month?
	11.	How much was credited last month?
	12.	How much did I receive in September 2023?
	13.	List all credits from “Payment Received” this year.
	14.	What was my largest single credit in 2024?

⸻

Debits (money out)
	15.	How much did I spend this month?
	16.	How much did I spend last month?
	17.	How much did I spend on fuel in 2023?
	18.	Which merchant did I spend the most at this year?
	19.	How much have I spent in August 2025?

⸻

Payments
	20.	What is my total payment amount this year?
	21.	How many payments did I make in July 2023?
	22.	What was the highest payment I made in 2024?
	23.	List all payment transactions with their dates.
	24.	How much was my last payment to Shell?

⸻

Category/merchant-specific
	25.	How much did I spend on dining in 2023?
	26.	Show me all transactions with Southwest Airlines.
	27.	Which merchants have I used most often this year?
	28.	How much did I spend at Amazon in August 2025?

⸻

Timeframe & filtering
	29.	How much money was credited between July 2023 and September 2023?
	30.	Show all transactions between 2023-07-01 and 2023-09-30.
	31.	What is my total debit amount for Q3 2023?