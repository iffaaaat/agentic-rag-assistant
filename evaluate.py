
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
            }

            print(f"  Expected: {expected}")
            print(f"  Predicted: {predicted}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

        except Exception as error:
            # Errors count as failed evaluation cases.
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

    print("\n========== EVALUATION REPORT ==========")
    print(f"Total Questions : {total}")
    print(f"Correct         : {correct}")
    print(f"Incorrect       : {total - correct}")
    print(f"Status Accuracy : {accuracy:.2f}%")
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
