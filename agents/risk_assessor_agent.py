"""Risk assessor agent — assesses portfolio risk and emits evaluation metrics."""
import boto3
import yaml
import os
from .base_agent import BaseFinanceAgent

SYSTEM_PROMPT = """You are a portfolio risk manager. Given a portfolio of holdings, assess:
overall_risk_level (low/medium/high/critical), concentration_risk, volatility_estimate,
recommended_actions (list), and risk_score (0-100). Be specific and actionable."""


class RiskAssessorAgent(BaseFinanceAgent):
    def __init__(self):
        super().__init__("RiskAssessor")
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self._eval_namespace = cfg["observability"]["metrics_namespace"] + "/Evaluations"

    def _run(self, input_data: dict) -> dict:
        portfolio = input_data.get("portfolio", [])
        total_value = input_data.get("total_value", 0)

        user_msg = (
            f"Portfolio holdings: {portfolio}\n"
            f"Total portfolio value: ${total_value:,}\n"
            "Assess the overall portfolio risk."
        )
        response, in_tok, out_tok = self._invoke_bedrock(SYSTEM_PROMPT, user_msg)

        # Emit evaluation quality scores to the Evaluations namespace
        self._emit_eval_scores()

        return {
            "agent": self.agent_name,
            "risk_report": response,
            "portfolio_value": total_value,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    def _emit_eval_scores(self):
        """Emit placeholder quality scores to AgentCore/Evaluations namespace."""
        scores = [
            ("CorrectnessScore", 0.88),
            ("HelpfulnessScore", 0.91),
            ("HarmlessnessScore", 0.97),
            ("TaskCompletionScore", 0.85),
        ]
        dims = [{"Name": "AgentName", "Value": self.agent_name}]
        metric_data = [
            {"MetricName": name, "Value": val, "Unit": "None", "Dimensions": dims}
            for name, val in scores
        ]
        try:
            self._cw.put_metric_data(Namespace=self._eval_namespace, MetricData=metric_data)
        except Exception:
            pass
