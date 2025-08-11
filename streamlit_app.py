import os, json, streamlit as st
from src.engine import ask_tx
from src.io import load_transactions
from src.semantic_index import build_index

st.set_page_config(page_title="TX Copilot (Semantic RAG)", page_icon="💳")
st.title("TX Copilot (Semantic RAG)")

with st.sidebar:
    st.text_input("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "", key="baseurl")
    st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "", type="password", key="key")
    use_llm = st.toggle("Use LLM", value=True)
    use_agent = st.toggle("Use Agent (LlamaIndex)", value=False)
    use_tools = st.toggle("Use LLM tools (function calling)", value=True)
    hist_n = st.number_input("History turns to send", min_value=0, max_value=20, value=8, step=1)
    if st.button("Apply"):
        if st.session_state.baseurl: os.environ["OPENAI_BASE_URL"] = st.session_state.baseurl
        if st.session_state.key: os.environ["OPENAI_API_KEY"] = st.session_state.key
        os.environ['USE_LLM_TOOLS'] = 'true' if use_tools else 'false'
        st.success("Applied.")
    st.divider()
    if st.button("Build semantic index"):
        try:
            tx = load_transactions()
            build_index(tx)
            st.success("Semantic index built ✅")
        except Exception as e:
            st.error(f"Index build failed: {e}")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None); st.rerun()

if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"]=="assistant" and m.get("is_json"): st.json(m["content"])
        else: st.markdown(m["content"] if isinstance(m["content"], str) else str(m["content"]))

q = st.chat_input("Ask about your transactions…")
if q:
    st.session_state.messages.append({"role":"user","content": q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Working…"):
            history = st.session_state.messages[-hist_n:] if hist_n>0 else []
            if use_agent:
                from src.agent_llamaindex import ask_agent
                res = ask_agent(q)
            else:
                res = ask_tx(q, use_llm=use_llm, chat_history=history)
            st.json(res)
            st.session_state.messages.append({"role":"assistant","content": res, "is_json": True})


with st.sidebar:
    st.divider()
    st.caption("Dataset tools")
    if st.button("Generate dataset"):
        import subprocess, sys, os
        try:
            out_dir = "data_synth"
            cmd = [sys.executable, "scripts/generate_tx_dataset.py", "--out-dir", out_dir, "--months", "3", "--accounts", "2", "--avg-per-month", "80", "--seed", "42", "--build-index"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            st.code(r.stdout or r.stderr)
            if r.returncode == 0:
                st.success("Dataset generated and indexed ✅")
        except Exception as e:
            st.error(f"Generation failed: {e}")


with st.sidebar:
    st.divider()
    show_ret = st.toggle("Show retrieval debug", value=False)

if show_ret and st.session_state.messages:
    try:
        from src.semantic_index import semantic_search
        q = st.session_state.messages[-1]["content"]
        ctx = semantic_search(q, top_k=8)
        st.subheader("Top matches")
        for d in ctx:
            st.write({
                "id": d.get("id"),
                "merchant": d.get("merchant"),
                "category": d.get("category"),
                "score": round(d.get("score", 0.0), 4),
                "text": d.get("text")[:180] + "…" if d.get("text") else "",
            })
    except Exception as e:
        st.warning(f"Retrieval debug failed: {e}")


with st.sidebar:
    st.divider()
    st.caption("Company domain glossary")
    use_gloss = st.toggle("Include glossary in prompt", value=True)
    if use_gloss: os.environ["USE_GLOSSARY_IN_PROMPT"] = "true"
    else: os.environ["USE_GLOSSARY_IN_PROMPT"] = "false"
    up = st.file_uploader("Upload company glossary (YAML)", type=["yaml","yml"])
    if up is not None:
        import yaml, os
        try:
            content = up.read().decode("utf-8")
            with open("data/domain_glossary.yaml","w",encoding="utf-8") as f:
                f.write(content)
            st.success("Glossary uploaded and saved.")
        except Exception as e:
            st.error(f"Failed to save glossary: {e}")
    if st.button("Preview glossary"):
        try:
            import yaml
            with open("data/domain_glossary.yaml","r",encoding="utf-8") as f:
                st.code(f.read())
        except Exception as e:
            st.error(str(e))


with st.sidebar:
    st.divider()
    st.caption("FAISS Vector Store")
    if st.button("Build FAISS index"):
        from src.io import load_transactions
        from src.faiss_index import build_faiss_index
        try:
            tx = load_transactions()
            idx_path, meta_path = build_faiss_index(tx)
            st.success(f"FAISS index built ✅\n{idx_path}\n{meta_path}")
        except Exception as e:
            st.error(f"FAISS build failed: {e}")
