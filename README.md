# Agentic RAG Assistant

A document question-answering application built with Python, Streamlit, OpenAI, LangChain, and FAISS.

The project explores a Retrieval-Augmented Generation (RAG) workflow with evidence evaluation, query rewriting, candidate-answer verification, and automated testing.

Its main objective is to improve the reliability of document-grounded answers by distinguishing directly supported information from missing information and unsupported assumptions.

## Features

- Upload and query `.txt` and `.docx` documents.
- Semantic document retrieval using OpenAI embeddings and FAISS.
- LLM-based routing between general knowledge and document retrieval.
- Document-grounded question answering.
- Structured evidence classification:
  - `ANSWERABLE` — sufficient evidence supports an answer.
  - `NOT_SPECIFIED` — the document explicitly states that the requested information is not provided.
  - `INSUFFICIENT` — the retrieved evidence cannot establish an answer.
- Evidence quotation validation against retrieved context.
- Candidate-answer verification to detect unsupported inferences.
- Automatic query rewriting and retrieval retry when evidence is insufficient.
- Source-referenced answer generation.
- Citation-number validation with one automatic correction attempt.
- Final-answer verification against cited evidence.
- Interactive Streamlit interface displaying reasoning steps and retrieved context.
- Automated evaluation with predefined test cases.
- Unit tests covering routing, citation retry, and final-answer verification.

## How It Works

1. **Document processing:** Uploaded `.txt` or `.docx` files are converted into text and split into chunks.
2. **Embedding and indexing:** Document chunks are embedded and stored in a FAISS vector store.
3. **Dynamic routing:** When a document is uploaded, an LLM-based router decides whether the question requires document retrieval or can be answered directly using general knowledge. Ambiguous questions default to retrieval. Without a document, the application uses direct answering.
4. **Retrieval:** For document-related questions, the application retrieves relevant chunks from the FAISS vector store.
5. **Evidence evaluation:** The retrieved context is classified as `ANSWERABLE`, `NOT_SPECIFIED`, or `INSUFFICIENT`.
6. **Candidate verification:** For `ANSWERABLE` results, the proposed answer is checked against its supporting evidence.
7. **Retrieval retry:** If the evidence is insufficient or candidate verification fails, the query is rewritten and retrieval is attempted once more.
8. **Answer generation:** An answer is generated using the retrieved context when sufficient evidence is available.
9. **Citation validation:** The application checks whether cited source numbers exist and attempts one regeneration if citations are invalid or missing.
10. **Final-answer verification:** The generated answer is checked against its cited evidence. If verification fails, the application abstains rather than returning the unsupported answer.

For clearly general questions, the `DIRECT` route generates an answer without document retrieval or document-grounding verification.

The router is designed to prioritize retrieval for document-related or ambiguous questions. However, because routing is LLM-based, incorrect decisions are still possible.

The system uses LLM-based verification, which can reduce certain unsupported inferences but does not guarantee factual correctness.


## Project Structure

```text
agentic-rag-assistant/
├── app.py
├── evaluate.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── evaluator.py
│   ├── citations.py
│   ├── llm_utils.py
│   ├── config.py
│   ├── file_processor.py
│   └── vectorstore.py
│
├── tests/
│   ├── test_cases.json
│   ├── test_citation_retry.py
│   ├── test_final_answer_verification.py
│   └── test_routing.py
│
├── sample_documents/
│   ├── ai_overview.txt
│   ├── company_policy.docx
│   └── Northstar_Labs_RAG_Test_Handbook.docx
│
├── evaluation_results/
│   ├── results.json
│   ├── results.csv
│   ├── results_phase3.json
│   └── results_phase3.csv
│
└── screenshots/
    ├── direct.png
    ├── retrieval.png
    └── retry.png
```


## Technologies

- Python
- Streamlit
- OpenAI API and embeddings
- LangChain
- FAISS
- python-dotenv
- docx2txt

## Installation

Clone the repository:

```bash
git clone https://github.com/iffaaaat/agentic-rag-assistant.git
cd agentic-rag-assistant
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Start the application:

```bash
python -m streamlit run app.py
```

An OpenAI API key with available API credits is required for embedding generation and LLM calls.


## Automated Evaluation

The project includes a fictional employee handbook designed for testing document retrieval, answerability, and unsupported inferences.

The evaluation dataset contains 20 questions covering:

- Directly answerable facts.
- Information explicitly identified as unavailable.
- Questions requiring abstention.
- Unsupported inferences involving company-wide technologies and specific systems.
- Distinctions between response targets and guaranteed resolution times.

Run the evaluation:

```bash
python evaluate.py --document sample_documents/Northstar_Labs_RAG_Test_Handbook.docx
```

Results are saved in `evaluation_results/`.

### Unit Tests

The project includes automated unit tests using Python's built-in `unittest` framework.

Run all tests from the project root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests cover:

- Successful correction of invalid citations.
- Safe rejection when citation regeneration fails.
- Acceptance of a verified final answer.
- Rejection of an unsupported final answer.
- Routing decisions between `DIRECT` and `RETRIEVE`.
- Conservative fallback for invalid routing responses.
- Direct-answer execution through the agent workflow.

These tests use mocked components and do not require live OpenAI API calls.

**Current result:** 9/9 unit tests passing.

### Evaluation Results

| Version | Correct classifications | Accuracy |
|---|---:|---:|
| Before recent reliability improvements | 16/20 | 80% |
| After evidence normalization and claim verification | 19/20 | 95% |

The latest recorded 20-question evaluation retained 95% classification accuracy after introducing citation validation and final-answer verification.

Of the 20 questions:

- 14 generated answers passed citation validation.
- 14 generated answers passed LLM-based final-answer verification.
- 6 questions were classified as `INSUFFICIENT` and did not proceed to answer generation.
- No validation-related abstentions occurred.

These validation results represent internal checks performed by the system, not independent guarantees of factual correctness.

The evaluation correctly rejected unsupported payroll-system assumptions while retaining supported answers about the engineering team's technologies.

**Dynamic routing update:** The application now supports LLM-based routing between general knowledge and document retrieval. The updated routing implementation passed 9 automated unit tests and 3 manually conducted live scenarios. However, the full 20-question benchmark has not yet been rerun after this routing change.

**Limitations:** This is a small, manually designed benchmark. Several test cases informed the development process, so the reported accuracy should not be interpreted as performance on an independent dataset. One test involving the CEO's salary has an ambiguous expected classification and requires review.

The evaluation measures classification agreement, not comprehensive final-answer factual correctness or citation accuracy.


### Evaluation Results

| Version | Correct classifications | Accuracy |
|---|---:|---:|
| Before recent reliability improvements | 16/20 | 80% |
| After evidence normalization and claim verification | 19/20 | 95% |

The latest 20-question evaluation retained 95% classification accuracy after
introducing citation validation and final-answer verification.

Of the 20 questions:
- 14 generated answers passed citation validation.
- 14 generated answers passed LLM-based final-answer verification.
- 6 questions were classified as INSUFFICIENT and did not proceed to answer generation.
- No validation-related abstentions occurred.

These validation results represent internal checks performed by the system,
not independent guarantees of factual correctness.

The latest evaluation correctly rejected unsupported payroll-system assumptions while retaining supported answers about the engineering team's technologies.

**Limitations:** This is a small, manually designed benchmark. Several test cases informed the development process, so the reported accuracy should not be interpreted as performance on an independent dataset. One test involving the CEO's salary has an ambiguous expected classification and requires review.

The evaluation measures classification agreement, not comprehensive final-answer factual correctness or citation accuracy.

## Example Workflows

### Document Retrieval

The application retrieves relevant information from uploaded documents and generates source-referenced answers.

![Document retrieval example](screenshots/retrieval.png)

### Direct Answer

When no document is available, the system can answer general questions directly.

![Direct answer example](screenshots/direct.png)

### Retrieval Retry

If the retrieved evidence is insufficient, or a proposed answer fails verification, the application rewrites the query and performs a second retrieval attempt.

![Retrieval retry example](screenshots/retry.png)

## Known Limitations and Future Improvements

- Citation validation confirms that referenced source numbers exist, but does not independently establish whether every cited source supports its associated claim.
- Final-answer verification uses an LLM to evaluate the answer against its combined cited evidence. It does not verify each claim-to-citation relationship separately.
- LLM-based verification may produce incorrect judgments and introduces additional API costs.
- Error handling can be improved to distinguish invalid model responses and API failures from genuinely insufficient evidence.
- Evaluate the system against additional unseen documents and questions to assess generalization.
- Add support for PDF and multi-document retrieval.
- Introduce persistent vector storage and optional conversation memory.
- Improve retrieval diagnostics and the Streamlit user interface.

## Project Purpose

This project is intended as an AI engineering portfolio demonstration of document retrieval, LLM orchestration, structured output validation, and evaluation-driven reliability improvements.

It is a learning and demonstration project rather than a production-ready document intelligence service.