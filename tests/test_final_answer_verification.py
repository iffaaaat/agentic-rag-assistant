import unittest
from unittest.mock import patch

from src.agent import run_agent


class TestFinalAnswerVerification(unittest.TestCase):

    def setUp(self):
        """Prepare shared test data."""
        self.fake_context = (
            "[Source 1]: Employees receive 16 days "
            "of annual leave per calendar year."
        )

        self.fake_evaluation = {
            "status": "ANSWERABLE",
            "reason": "Evidence supports 16 days.",
            "evidence": (
                "Employees receive 16 days "
                "of annual leave per calendar year."
            ),
            "candidate_answer": "16 days",
        }

    def test_unsupported_final_answer_is_rejected(self):
        """Reject an unsupported answer even with a valid citation."""

        with patch(
            "src.agent.route_question",
            return_value="RETRIEVE",
        ), patch(
            "src.agent.retrieve_context",
            return_value=(self.fake_context, 0.5),
        ), patch(
            "src.agent.evaluate_and_verify",
            return_value=self.fake_evaluation,
        ), patch(
            "src.agent.generate_answer",
            return_value="Employees receive 20 days of leave [Source 1].",
        ), patch(
            "src.agent.verify_final_answer",
            return_value={
                "supported": False,
                "reason": "The source specifies 16 days, not 20.",
            },
        ) as verifier_mock:

            answer, steps, context, distance, status = run_agent(
                query="How many days of annual leave do employees receive?",
                vectorstore=object(),
                document_text=self.fake_context,
            )

        self.assertEqual(status, "ANSWERABLE")
        self.assertIn("Decision: RETRIEVE", steps)
        self.assertIn("Citation Validation: PASSED", steps)
        self.assertIn("Final Answer Verification: FAILED", steps)
        self.assertEqual(verifier_mock.call_count, 1)
        self.assertIn("I couldn't verify the final answer", answer)
        self.assertNotIn("20 days", answer)

    def test_supported_final_answer_is_accepted(self):
        """Accept an answer supported by its cited evidence."""

        with patch(
            "src.agent.route_question",
            return_value="RETRIEVE",
        ), patch(
            "src.agent.retrieve_context",
            return_value=(self.fake_context, 0.5),
        ), patch(
            "src.agent.evaluate_and_verify",
            return_value=self.fake_evaluation,
        ), patch(
            "src.agent.generate_answer",
            return_value="Employees receive 16 days of leave [Source 1].",
        ), patch(
            "src.agent.verify_final_answer",
            return_value={
                "supported": True,
                "reason": "The source explicitly supports 16 days.",
            },
        ) as verifier_mock:

            answer, steps, context, distance, status = run_agent(
                query="How many days of annual leave do employees receive?",
                vectorstore=object(),
                document_text=self.fake_context,
            )

        self.assertEqual(status, "ANSWERABLE")
        self.assertIn("Decision: RETRIEVE", steps)
        self.assertIn("Citation Validation: PASSED", steps)
        self.assertIn("Final Answer Verification: PASSED", steps)
        self.assertEqual(verifier_mock.call_count, 1)
        self.assertIn("16 days", answer)
        self.assertIn("[Source 1]", answer)


if __name__ == "__main__":
    unittest.main()
