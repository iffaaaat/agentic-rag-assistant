import argparse
import csv
import json
from pathlib import Path

import docx2txt

from src.agent import run_agent
from src.vectorstore import create_vectorstore


def load_test_cases(path):
    """Load and validate the evaluation dataset."""
    with open(path, "r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("Test dataset must be a non-empty list.")

    allowed = {
        "ANSWERABLE",
        "NOT_SPECIFIED",
        "INSUFFICIENT",
    }

    for case in cases:
        if case["expected_status"] not in allowed:
            raise ValueError(
                f"Invalid expected status in test {case['id']}"
            )

    return cases


def load_document(path):
    """Extract text from the test DOCX file."""
    text = docx2txt.process(str(path))

    if not text or not text.strip():
        raise ValueError("Document contains no readable text.")

    return text


def run_evaluation(document_path, test_path):
    """Run all questions and measure classification accuracy."""

    print("\nLoading test document...")
    document_text = load_document(document_path)

    print("Creating FAISS vector store...")
    vectorstore = create_vectorstore(document_text)

    test_cases = load_test_cases(test_path)

    results = []
    correct = 0

    print(f"\nRunning {len(test_cases)} evaluation cases...\n")

    for index, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected = case["expected_status"]

        print(f"[{index}/{len(test_cases)}] {question}")

        try:
            answer, steps, context, distance, predicted = (
                run_agent(
                    query=question,
                    vectorstore=vectorstore,
                    document_text=document_text,
                )
            )

            passed = predicted == expected

            if passed:
                correct += 1

            citation_status = "NOT_RUN"
            final_answer_status = "NOT_RUN"

            for step in steps:
                if step == "Citation Validation: PASSED":
                    citation_status = "PASSED"

                elif step == "Citation Validation: FAILED":
                    citation_status = "FAILED"

                elif step == "Citation Retry: PASSED":
                    citation_status = "PASSED_AFTER_RETRY"

                elif step == "Citation Retry: FAILED":
                    citation_status = "FAILED_AFTER_RETRY"

                elif step == "Final Answer Verification: PASSED":
                    final_answer_status = "PASSED"

                elif step == "Final Answer Verification: FAILED":
                    final_answer_status = "FAILED"

            answer_abstained = (
                "I couldn't produce an answer with valid source "
                "citations" in answer
                or "I couldn't verify the final answer" in answer
            )

            result = {
                "id": case["id"],
                "question": question,
                "expected_status": expected,
                "predicted_status": predicted,
                "passed": passed,
                "expected_answer": case["expected_answer"],
                "actual_answer": answer,
                "distance": distance,
                "steps": steps,
                "context": context,
                "error": "",
                "citation_status": citation_status,
                "final_answer_status": final_answer_status,
                "answer_abstained": answer_abstained,
            }

            print(f"  Expected: {expected}")
            print(f"  Predicted: {predicted}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            print(f"  Citation: {citation_status}")
            print(f"  Final Answer: {final_answer_status}")
            print(f"  Validation Abstained: {answer_abstained}")

        except Exception as error:
            result = {
                "id": case["id"],
                "question": question,
                "expected_status": expected,
                "predicted_status": "ERROR",
                "passed": False,
                "expected_answer": case["expected_answer"],
                "actual_answer": "",
                "distance": None,
                "steps": [],
                "context": "",
                "error": str(error),
            }

            print(f"  ERROR: {error}")

        results.append(result)
        print("-" * 60)

    total = len(test_cases)
    accuracy = correct / total * 100
    
    citation_counts = {
        "PASSED": 0,
        "FAILED": 0,
        "PASSED_AFTER_RETRY": 0,
        "FAILED_AFTER_RETRY": 0,
        "NOT_RUN": 0,
    }

    final_answer_counts = {
        "PASSED": 0,
        "FAILED": 0,
        "NOT_RUN": 0,
    }

    validation_abstentions = 0

    for result in results:
        citation_status = result.get("citation_status", "NOT_RUN")
        final_status = result.get("final_answer_status", "NOT_RUN")

        citation_counts[citation_status] += 1
        final_answer_counts[final_status] += 1

        if result.get("answer_abstained", False):
            validation_abstentions += 1


    
    print("\n========== EVALUATION REPORT ==========")
    print(f"Total Questions       : {total}")
    print(f"Correct Classifications: {correct}")
    print(f"Incorrect             : {total - correct}")
    print(f"Status Accuracy       : {accuracy:.2f}%")

    print("\n--- Citation Validation ---")
    for status, count in citation_counts.items():
        print(f"{status:20}: {count}")

    print("\n--- Final Answer Verification ---")
    for status, count in final_answer_counts.items():
        print(f"{status:20}: {count}")

    print("\n--- Answer Reliability ---")
    print(f"Validation Abstentions: {validation_abstentions}")

    print("=======================================\n")


    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    columns = [
        "id",
        "question",
        "expected_status",
        "predicted_status",
        "passed",
        "expected_answer",
        "actual_answer",
        "distance",
        "error",
        "citation_status",
        "final_answer_status",
        "answer_abstained",
    ]

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                column: result.get(column)
                for column in columns
            })

    print(f"JSON report: {json_path}")
    print(f"CSV report:  {csv_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Agentic RAG classification accuracy."
    )

    parser.add_argument(
        "--document",
        required=True,
        help="Path to the DOCX test document.",
    )

    parser.add_argument(
        "--tests",
        default="tests/test_cases.json",
        help="Path to the evaluation dataset.",
    )

    args = parser.parse_args()

    run_evaluation(
        document_path=args.document,
        test_path=args.tests,
    )
