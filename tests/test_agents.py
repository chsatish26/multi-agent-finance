"""
Unit tests for finance agents — mocks Bedrock and CloudWatch to avoid AWS calls.
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mock_bedrock_response(text="Test response", in_tok=10, out_tok=20):
    import json
    body_bytes = json.dumps({
        "content": [{"text": text}],
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }).encode()
    mock_resp = MagicMock()
    mock_resp["body"].read.return_value = body_bytes
    return mock_resp


class TestBaseAgent(unittest.TestCase):
    @patch("boto3.client")
    def test_invoke_emits_metrics(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response("ok", 5, 10)
        mock_cw = MagicMock()
        mock_logs = MagicMock()
        mock_logs.create_log_stream.return_value = {}
        mock_boto.side_effect = lambda svc, **kw: {
            "bedrock-runtime": mock_bedrock,
            "cloudwatch": mock_cw,
            "logs": mock_logs,
        }[svc]

        from agents.stock_analyst_agent import StockAnalystAgent
        agent = StockAnalystAgent()
        result = agent.invoke({"ticker": "AAPL"})

        self.assertEqual(result["ticker"], "AAPL")
        self.assertIn("analysis", result)
        # RuntimeInvocations + RuntimeLatency + InputTokens + OutputTokens = 4 calls
        self.assertGreaterEqual(mock_cw.put_metric_data.call_count, 4)


class TestStockAnalystAgent(unittest.TestCase):
    @patch("boto3.client")
    def test_run_returns_analysis(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response("Bullish on TSLA")
        mock_boto.side_effect = lambda svc, **kw: {
            "bedrock-runtime": mock_bedrock,
            "cloudwatch": MagicMock(),
            "logs": MagicMock(),
        }[svc]

        from agents.stock_analyst_agent import StockAnalystAgent
        agent = StockAnalystAgent()
        result = agent._run({"ticker": "TSLA", "context": "Strong EV demand"})
        self.assertEqual(result["ticker"], "TSLA")
        self.assertIn("Bullish", result["analysis"])


class TestCreditEvaluatorAgent(unittest.TestCase):
    @patch("boto3.client")
    def test_run_returns_evaluation(self, mock_boto):
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = _mock_bedrock_response("Low risk. Approve.")
        mock_boto.side_effect = lambda svc, **kw: {
            "bedrock-runtime": mock_bedrock,
            "cloudwatch": MagicMock(),
            "logs": MagicMock(),
        }[svc]

        from agents.credit_evaluator_agent import CreditEvaluatorAgent
        agent = CreditEvaluatorAgent()
        result = agent._run({"applicant": {"credit_score": 750}, "loan_amount": 100000})
        self.assertIn("evaluation", result)
        self.assertEqual(result["loan_amount"], 100000)


if __name__ == "__main__":
    unittest.main()
