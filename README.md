# Banking Copilot with FAISS Flat Index

This is a production-ready **banking copilot** that answers questions using **transaction data** with
**FAISS Flat (exact cosine) retrieval** and Llama/BGE embeddings.

## Key Features
- **Exact retrieval** via FAISS Flat index (IndexFlatIP) on normalized embeddings
- **Disk persistence** for the vector index (`index_faiss/`)
- **Streamlit chat UI** with conversation history
- **Context engineering** — no fine-tuning needed
- **RAG pipeline** using Llama as the LLM and BGE for embeddings
- **Dataset generator** for synthetic transaction data

## Why FAISS Flat?
- Returns **exact** nearest neighbors (no approximation)
- Perfect for small/medium datasets (thousands of transactions)
- Simpler and lighter than HNSW or IVF indexes
- No tuning parameters — just works

## Requirements
```bash
pip install -r requirements.txt
```
You'll need:
- `faiss-cpu`
- `llama-index` (for retrieval orchestration)
- `streamlit`

## Usage
1. **Set environment variables**:
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="your_key"
export CHAT_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export EMBED_MODEL="BAAI/bge-en-icl"
```
(These can point to any OpenAI-compatible endpoint, including local models.)

2. **Build the FAISS index**:
```bash
python scripts/build_faiss_index.py --transactions data/transactions.json
```
This creates `index_faiss/tx_faiss.index` and `tx_faiss.meta.json`.

3. **Run the app**:
```bash
streamlit run streamlit_app.py
```

4. **Ask questions**:
- "How many PURCHASE transactions over $200 last month?"
- "List all installment payments in the last 6 months."
- "Show interest charges in my statement."

## Project Structure
```
src/
  faiss_index.py      # FAISS Flat index build/load/search
  retrieval.py        # RAG retrieval pipeline
  engine.py           # LLM + retriever integration
  tools.py            # Domain-specific tools
scripts/
  build_faiss_index.py # CLI to build the FAISS index
data/
  transactions.json    # Example dataset
  domain_glossary.yaml # Optional domain glossary
streamlit_app.py       # Main Streamlit UI
```

## Notes
- Uses **exact** cosine similarity (inner product on L2-normalized vectors)
- Fast enough for datasets in the thousands–tens of thousands of transactions
- Switch to HNSW or IVF if you scale to millions of records
