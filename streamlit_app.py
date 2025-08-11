import os, json, streamlit as st
from src.engine import ask_tx

st.set_page_config(page_title="TX Copilot (Prod-ready)", page_icon="💳")
st.title("TX Copilot (Prod-ready)")

with st.sidebar:
    st.text_input("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "", key="baseurl")
    st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "", type="password", key="key")
    use_llm = st.toggle("Use LLM", value=True)
    hist_n = st.number_input("History turns to send", min_value=0, max_value=20, value=8, step=1)
    if st.button("Apply"):
        if st.session_state.baseurl: os.environ["OPENAI_BASE_URL"] = st.session_state.baseurl
        if st.session_state.key: os.environ["OPENAI_API_KEY"] = st.session_state.key
        st.success("Applied.")
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
            res = ask_tx(q, use_llm=use_llm, chat_history=history)
            st.json(res)
            st.session_state.messages.append({"role":"assistant","content": res, "is_json": True})
