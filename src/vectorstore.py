
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    TOP_K,
)


def create_vectorstore(text):
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return FAISS.from_documents(docs, embeddings)


def retrieve_context(vectorstore, query, k=TOP_K):
    docs_and_scores = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    if not docs_and_scores:
        return None, None

    context = "\n\n".join(
        f"[Source {i + 1}]: {doc.page_content}"
        for i, (doc, _) in enumerate(docs_and_scores)
    )

    # FAISS returns a distance, not a confidence probability.
    # With this index configuration, smaller distances indicate
    # closer matches.
    best_distance = min(
        float(score)
        for _, score in docs_and_scores
    )

    return context, best_distance
