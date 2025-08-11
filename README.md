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
