import streamlit as st

from src.agent import run_agent
from src.file_processor import process_file
from src.vectorstore import create_vectorstore


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic RAG Assistant")
st.caption("An AI-powered document question answering system using Agentic RAG.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "query_input" not in st.session_state:
    st.session_state.query_input = ""


def clear_input():
    """Clear the question input field."""
    st.session_state.query_input = ""


uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=["txt", "docx"],
)

if uploaded_file:

    try:
        text = process_file(uploaded_file)

        st.session_state.vectorstore = create_vectorstore(text)

        st.success("✅ Document processed and indexed successfully!")

    except Exception as e:
        st.error(str(e))

query = st.text_input(
    "💬 Ask a Question",
    key="query_input",
)

st.button(
    "Clear Input",
    on_click=clear_input,
)

if query:

    with st.spinner("Agent is thinking..."):

        answer, steps, context, confidence = run_agent(
            query,
            st.session_state.vectorstore,
        )

    st.subheader("🧠 Agent Reasoning")

    for step in steps:
        st.write(step)

    if confidence is not None:
        st.metric(
            "Retrieval Confidence",
            confidence,
        )

    if context:
        st.subheader("Retrieved Context")
        st.text_area(
            "📚 Retrieved Context",
            context,
            height=250,
)

    st.divider()

    st.subheader("Final Answer")

    st.write(answer)