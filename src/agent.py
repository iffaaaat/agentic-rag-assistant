from src.config import (
    CHAT_MODEL,
    client,
)

from src.vectorstore import retrieve_context


def evaluate_retrieval(query, document_text=None, context=None):

    prompt = f"""
You are an AI agent responsible for deciding how to answer a user question.

Your task has two possible stages:

1. Decide whether the question should use the uploaded document:
   - RETRIEVE
   - DIRECT

2. If retrieved context is provided, evaluate whether it contains
   enough information to answer the question:
   - GOOD
   - BAD

Uploaded document:
{document_text if document_text else "No document uploaded."}

Question:
{query}

Retrieved context:
{context if context else "None"}

Routing rules:

- Choose RETRIEVE if the uploaded document contains information that
  could reasonably answer the question.

- Choose DIRECT if the question is clearly unrelated to the uploaded
  document and can be answered without information from it.

- Do not choose RETRIEVE merely because a document exists.

Context evaluation rules:

- GOOD = the retrieved context contains enough information to answer
  the question.

- BAD = the context is related to the topic but does not contain
  enough information to answer the question.

- Do not mark context as GOOD merely because it discusses a similar topic.

Examples:

Document:
"Employees must submit annual leave requests at least five working
days in advance."

Question:
"When should I submit annual leave?"
Decision: RETRIEVE

Question:
"What is the capital of Japan?"
Decision: DIRECT

Question:
"What should I do if IT support is unavailable outside business hours?"
Decision: RETRIEVE

Output exactly:

Decision: RETRIEVE or DIRECT
Evaluation: GOOD or BAD or NONE
"""

    response  = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content.strip()

    decision = "RETRIEVE"
    evaluation = "NONE"

    for line in output.splitlines():

        if line.upper().startswith("DECISION"):
            decision = line.split(":")[-1].strip().upper()

        elif line.upper().startswith("EVALUATION"):
            evaluation = line.split(":")[-1].strip().upper()

    return decision, evaluation


def rewrite_query(query):

    prompt = f"""
Rewrite this question to improve document retrieval.

Rules:
- Make it more specific
- Include key terms likely in documents
- Keep original meaning

Original Question:
{query}

Improved Query:
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def generate_answer(query, context=None):

    if context:

        prompt = f"""
Answer the question using ONLY the sources below.

If the answer is not contained, say "No Relevant Information Found".

Cite like [Source 1].

{context}

Question:
{query}
"""
    else:
        prompt = f"""
Answer the question directly.

Question:
{query}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def run_agent(query, vectorstore, document_text=None):

    steps_log = []

    if vectorstore:
        decision, _ = evaluate_retrieval(
            query,
            document_text=document_text,
        )
    else:
        decision = "DIRECT"

    steps_log.append(f"Decision: {decision}")

    if decision == "DIRECT" or not vectorstore:

        answer = generate_answer(query)

        return answer, steps_log, None, None

    context, confidence = retrieve_context(
        vectorstore,
        query,
    )

    _, evaluation = evaluate_retrieval(
        query,
        document_text=document_text,
        context=context,
    )

    steps_log.append(
        f"Evaluation: {evaluation}"
    )

    if evaluation == "BAD":

        steps_log.append(
            "Retrying with improved query..."
        )

        new_query = rewrite_query(query)

        steps_log.append(
            f"New Query: {new_query}"
        )

        context, confidence = retrieve_context(
            vectorstore,
            new_query,
        )

        _, evaluation = evaluate_retrieval(
            query,
            document_text=document_text,
            context=context,
        )

        steps_log.append(
            f"Updated Retrieval Similarity: {confidence:.2f}"
        )

        steps_log.append(
            f"Re-Evaluation: {evaluation}"
        )

    if evaluation == "BAD":

        return (
            "I could not confidently find relevant information in the document after retrying.",
            steps_log,
            context,
            confidence,
        )

    if not context:

        return (
            "I could not find relevant information in the document.",
            steps_log,
            context,
            confidence,
        )

    answer = generate_answer(
        query,
        context,
    )

    return (
        answer,
        steps_log,
        context,
        confidence,
    )