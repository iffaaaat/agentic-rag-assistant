# 🤖 Agentic RAG Assistant

An AI-powered document question answering application built with OpenAI, LangChain, FAISS, and Streamlit.

## Features

- Upload TXT and DOCX documents
- Retrieval-Augmented Generation (RAG)
- Agent-based decision making (Direct Answer vs Retrieval)
- Query rewriting for improved retrieval
- Retrieval confidence scoring
- Context evaluation before answer generation
- Interactive Streamlit interface

## Workflow

1. Upload a document.
2. The document is split into chunks and embedded into a FAISS vector database.
3. The agent decides whether retrieval is required.
4. If retrieval is needed, relevant context is retrieved.
5. The agent evaluates the retrieved context.
6. If confidence is low, the query is rewritten and retrieval is attempted again.
7. The final answer is generated using the retrieved context.

## Tech Stack

- Python
- Streamlit
- OpenAI API
- LangChain
- FAISS
- python-docx
- docx2txt


## License

MIT