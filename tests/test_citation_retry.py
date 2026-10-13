import unittest
from unittest.mock import patch

from src.agent import run_agent


class TestCitationRetry(unittest.TestCase):

    def setUp(self):
        """Prepare shared test data."""
        self.fake_context = (
            "[Source 1]: The engineering team uses Python "
            "for backend services."
        )

        self.fake_evaluation = {
            "status": "ANSWERABLE",
            "reason": "The evidence supports Python.",
            "evidence": (
                "The engineering team uses Python "
                "for backend services."
            ),
            "candidate_answer": "Python",
        }

    def test_invalid_citation_is_corrected(self):
        """An invalid citation should trigger one successful retry."""

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
            return_value="The team uses Python [Source 7].",
        ), patch(
            "src.agent.retry_answer_with_citations",
            return_value="The team uses Python [Source 1].",
        ) as retry_mock, patch(
            "src.agent.verify_final_answer",
            return_value={
                "supported": True,
                "reason": "The source supports Python.",
            },
        ):

            answer, steps, context, distance, status = run_agent(
                query="What language does engineering use?",
                vectorstore=object(),
                document_text=self.fake_context,
            )

        self.assertEqual(status, "ANSWERABLE")
        self.assertIn("Decision: RETRIEVE", steps)
        self.assertIn("Citation Validation: FAILED", steps)
        self.assertIn("Citation Retry: PASSED", steps)
        self.assertIn("Final Answer Verification: PASSED", steps)
        self.assertEqual(retry_mock.call_count, 1)
        self.assertIn("[Source 1]", answer)

    def test_invalid_citation_after_retry_is_rejected(self):
        """Reject an answer if citation regeneration still fails."""

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
            return_value="The team uses Python [Source 7].",
        ), patch(
            "src.agent.retry_answer_with_citations",
            return_value="The team uses Python [Source 9].",
        ) as retry_mock, patch(
            "src.agent.verify_final_answer",
        ) as verifier_mock:

            answer, steps, context, distance, status = run_agent(
                query="What language does engineering use?",
                vectorstore=object(),
                document_text=self.fake_context,
            )

        self.assertEqual(status, "ANSWERABLE")
        self.assertIn("Decision: RETRIEVE", steps)
        self.assertIn("Citation Validation: FAILED", steps)
        self.assertIn("Citation Retry: FAILED", steps)
        self.assertEqual(retry_mock.call_count, 1)

        verifier_mock.assert_not_called()

        self.assertIn("I couldn't produce an answer", answer)
        self.assertNotIn("[Source 9]", answer)


if __name__ == "__main__":
    unittest.main()
