
import unittest
from unittest.mock import patch

from src.agent import route_question, run_agent


class TestRouting(unittest.TestCase):

    def test_no_document_uses_direct(self):
        self.assertEqual(
            route_question("What is Python?", None),
            "DIRECT",
        )

    @patch("src.agent.ask_for_json")
    def test_document_question_uses_retrieval(self, mock_llm):
        mock_llm.return_value = {"decision": "RETRIEVE"}

        result = route_question(
            "What is the annual leave policy?",
            "Employee handbook with annual leave policies.",
        )

        self.assertEqual(result, "RETRIEVE")

    @patch("src.agent.ask_for_json")
    def test_general_question_uses_direct(self, mock_llm):
        mock_llm.return_value = {"decision": "DIRECT"}

        result = route_question(
            "What is the capital of Japan?",
            "Employee handbook about company policies.",
        )

        self.assertEqual(result, "DIRECT")

    @patch("src.agent.ask_for_json")
    def test_invalid_decision_defaults_to_retrieval(self, mock_llm):
        mock_llm.return_value = {"decision": "UNKNOWN"}

        result = route_question(
            "Who is the CEO?",
            "Employee handbook.",
        )

        self.assertEqual(result, "RETRIEVE")

    @patch("src.agent.generate_answer")
    @patch("src.agent.route_question")
    def test_run_agent_direct_route(self, mock_router, mock_answer):
        mock_router.return_value = "DIRECT"
        mock_answer.return_value = "Tokyo."

        answer, steps, context, distance, status = run_agent(
            query="What is the capital of Japan?",
            vectorstore=object(),
            document_text="Employee handbook.",
        )

        self.assertEqual(answer, "Tokyo.")
        self.assertIn("Decision: DIRECT", steps)
        self.assertIsNone(context)
        self.assertIsNone(status)

    @patch("src.agent.ask_for_json")
    def test_payroll_question_routes_to_retrieval(self, mock_llm):
        """Organization-specific payroll questions should use retrieval."""
        mock_llm.return_value = {"decision": "RETRIEVE"}

        decision = route_question(
            "Does Northstar Labs use Python for its payroll system?",
            (
                "Northstar Labs employee handbook. "
                "The engineering team uses Python for backend services."
            ),
        )

        self.assertEqual(decision, "RETRIEVE")


if __name__ == "__main__":
    unittest.main()
