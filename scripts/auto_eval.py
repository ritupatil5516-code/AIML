#!/usr/bin/env python3
# scripts/auto_eval.py

from __future__ import annotations

import os, json, csv, pathlib
from typing import List, Dict, Any

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)

import faiss


# --------------------------
# Config & paths
# --------------------------
DATA_DIR        = os.getenv("DATA_DIR", "data")
TX_PATH         = os.path.join(DATA_DIR, "transactions.json")
ACCT_PATH       = os.path.join(DATA_DIR, "account-summary.json")
REPORT_DIR      = os.getenv("REPORT_DIR", "reports")
INDEX_DIR       = os.getenv("INDEX_DIR", "indexes_li")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K           = int(os.getenv("EVAL_TOP_K", "4"))
NUM_QUESTIONS   = int(os.getenv("EVAL_NUM_QUESTIONS", "40"))  # tweak as needed


# --------------------------
# Data loading helpers
# --------------------------
def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_transactions() -> List[Dict[str, Any]]:
    data = _load_json(TX_PATH)
    return data.get("transactions", data)

def load_account_summaries() -> List[Dict[str, Any]]:
    data = _load_json(ACCT_PATH)
    return data.get("accounts", data)


# --------------------------
# Pack docs (compact, fielded)
# --------------------------
TX_FIELDS = (
    "transactionId transactionType transactionStatus transactionDateTime "
    "amount currencyCode endingBalance merchantName accountId debitCreditIndicator".split()
)

ACCT_FIELDS = [
    "accountId","accountNumberLast4","accountStatus","accountType","productType",
    "currentBalance","currentAdjustedBalance","totalBalance","availableCredit","creditLimit",
    "minimumDueAmount","pastDueAmount","highestPriorityStatus","balanceStatus",
    "paymentDueDate","paymentDueDateTime","billingCycleOpenDateTime","billingCycleCloseDateTime",
    "lastUpdatedDate","openedDate","closedDate","currencyCode","flags","subStatuses",
]

def _norm(v):
    if v is None: return ""
    if isinstance(v, (list, tuple)): return ", ".join(map(_norm, v))
    return str(v)

def tx_text(d: Dict[str, Any]) -> str:
    return " | ".join(f"{k}:{_norm(d.get(k))}" for k in TX_FIELDS if d.get(k) is not None)

def acct_text(d: Dict[str, Any]) -> str:
    return " | ".join(f"{k}:{_norm(d.get(k))}" for k in ACCT_FIELDS if d.get(k) is not None)


# --------------------------
# Build index (FAISS + BGE)
# --------------------------
def build_index(documents: List[Document]) -> VectorStoreIndex:
    # Models
    Settings.llm = OpenAI(model=CHAT_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # FAISS store (flat IP; embeddings are normalized by HF model)
    dim = Settings.embed_model.embed("test").shape[0]
    index = faiss.IndexFlatIP(dim)
    vs = FaissVectorStore(faiss_index=index)

    storage_context = StorageContext.from_defaults(vector_store=vs, persist_dir=INDEX_DIR)
    idx = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    storage_context.persist(persist_dir=INDEX_DIR)
    return idx


# --------------------------
# Make Documents
# --------------------------
def build_documents() -> List[Document]:
    tx = load_transactions()
    ac = load_account_summaries()

    docs: List[Document] = []
    # Transactions
    for t in tx:
        tid = t.get("transactionId") or t.get("id")
        if not tid:
            continue
        docs.append(Document(
            text=tx_text(t),
            metadata={"type":"transaction","transactionId":tid, "accountId":t.get("accountId")}
        ))
    # Accounts
    for a in ac:
        aid = a.get("accountId") or a.get("id")
        if not aid:
            continue
        docs.append(Document(
            text=acct_text(a),
            metadata={"type":"account","accountId":aid}
        ))
    return docs


# --------------------------
# Generate Q/A dataset
# --------------------------
def generate_dataset(documents: List[Document], num_questions: int = NUM_QUESTIONS):
    gen = RagDatasetGenerator.from_documents(
        documents=documents,
        llm=Settings.llm,
        num_questions=num_questions,
        show_progress=True,
    )
    dataset = gen.generate_dataset()
    # dataset.queries, dataset.relevant_docs (dict q->contexts), dataset.answers
    return dataset


# --------------------------
# Evaluate with Faithfulness + Relevancy
# --------------------------
def run_evaluation(index: VectorStoreIndex, dataset) -> Dict[str, Any]:
    qe = index.as_query_engine(similarity_top_k=TOP_K)

    faith = FaithfulnessEvaluator(llm=Settings.llm)
    relev = RelevancyEvaluator(llm=Settings.llm)

    runner = BatchEvalRunner(
        {"faithfulness": faith, "relevancy": relev},
        show_progress=True,
    )

    results = runner.run_with_query_engine(
        query_engine=qe,
        dataset=dataset,  # RagDataset with queries/contexts/answers
    )
    return results


# --------------------------
# Reporting
# --------------------------
def summarize_and_write(results: Dict[str, Any], out_dir: str = REPORT_DIR):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # results is a dict: metric -> List[EvaluationResult]
    rows = []
    for metric, eval_list in results.items():
        for ev in eval_list:
            rows.append({
                "metric": metric,
                "query": ev.query,
                "score": getattr(ev, "score", None),
                "passing": getattr(ev, "passing", None),
                "feedback": getattr(ev, "feedback", None),
                "response": getattr(ev, "response", None),
            })

    # JSON
    json_path = os.path.join(out_dir, "rag_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    # CSV
    csv_path = os.path.join(out_dir, "rag_eval_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric","query","score","passing","feedback","response"])
        w.writeheader(); w.writerows(rows)

    # Print quick summary
    def _avg(scores):
        vals = [r["score"] for r in rows if r["metric"]=="faithfulness" and isinstance(r["score"], (int,float))]
        return round(sum(vals)/max(len(vals),1), 3)
    faith_scores = [r for r in rows if r["metric"]=="faithfulness"]
    relev_scores = [r for r in rows if r["metric"]=="relevancy"]
    faith_pass = sum(1 for r in faith_scores if r["passing"])
    relev_pass = sum(1 for r in relev_scores if r["passing"])

    print(f"\nSaved: {json_path}\nSaved: {csv_path}")
    print(f"Faithfulness: {faith_pass}/{len(faith_scores)} passing (avg score≈{_avg(rows)})")
    print(f"Relevancy:    {relev_pass}/{len(relev_scores)} passing")


# --------------------------
# Main
# --------------------------
def main():
    print("Loading docs...")
    docs = build_documents()
    print(f"Docs: {len(docs)}")

    print("Building FAISS index...")
    idx = build_index(docs)

    print(f"Generating {NUM_QUESTIONS} questions...")
    dataset = generate_dataset(docs, NUM_QUESTIONS)

    print("Running evaluation (Faithfulness + Relevancy)...")
    results = run_evaluation(idx, dataset)

    summarize_and_write(results, REPORT_DIR)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    main()