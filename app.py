import streamlit as st
import tempfile
import os
from rag_engine import build_rag_chain

st.set_page_config(
    page_title="Medical Document Q&A",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Document Q&A")
st.markdown("Upload a medical PDF and ask questions. Answers are grounded in the document.")

uploaded_file = st.file_uploader("Upload Medical PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        chain = build_rag_chain(tmp_path)
        st.session_state.chain = chain
        st.session_state.ready = True

    st.success("Document ready. Ask your questions below.")

if "ready" in st.session_state and st.session_state.ready:
    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Searching document..."):
            result = st.session_state.chain.invoke(question)

        st.markdown("### Answer")
        st.write(result)
