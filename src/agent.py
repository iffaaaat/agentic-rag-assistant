from src.config import (
    CHAT_MODEL,
    client,
)

from src.vectorstore import retrieve_context


def evaluate_retrieval(query, document_uploaded=True, context=None):

    prompt = f"""
You are an AI agent.

Task:
1. Decide if retrieval is needed (RETRIEVE or DIRECT)
2. If context is provided, evaluate it (GOOD or BAD)

Rules:

Document uploaded:
{"Yes" if document_uploaded else "No"}

- If a document has been uploaded and the question could reasonably
  be answered from that document, choose RETRIEVE.

- Use DIRECT only if the question is clearly unrelated to the uploaded document.

Examples:

Uploaded policy document:
"What are working hours?" -> RETRIEVE

Uploaded employee handbook:
"When should I submit leave?" -> RETRIEVE

"What is the capital of Japan?" -> DIRECT

Question:
{query}

Context:
{context if context else "None"}

Output format:
Decision: RETRIEVE or DIRECT
Evaluation: GOOD or BAD (if context exists, else NONE)
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


def run_agent(query, vectorstore):

    steps_log = []

    if vectorstore:
        decision, _ = evaluate_retrieval(
            query,
            document_uploaded=True,
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
        document_uploaded=True,
        context=context,
    )

    steps_log.append(f"Evaluation: {evaluation}")

    steps_log.append(
        f"Confidence Score: {confidence}"
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
            document_uploaded=True,
            context=context,
        )

        steps_log.append(
            f"Updated Confidence: {confidence}"
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