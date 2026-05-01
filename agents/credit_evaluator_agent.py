"""Credit evaluator agent — evaluates credit risk for a loan application."""
from .base_agent import BaseFinanceAgent

SYSTEM_PROMPT = """You are a credit risk analyst. Given applicant financial data, evaluate credit risk.
Return: risk_score (0-100, higher = riskier), risk_tier (low/medium/high/very_high),
key_risk_factors (list), and recommendation (approve/conditional/decline) with brief rationale."""


class CreditEvaluatorAgent(BaseFinanceAgent):
    def __init__(self):
        super().__init__("CreditEvaluator")

    def _run(self, input_data: dict) -> dict:
        applicant = input_data.get("applicant", {})
        loan_amount = input_data.get("loan_amount", 0)

        user_msg = (
            f"Loan application details:\n"
            f"- Applicant: {applicant}\n"
            f"- Requested loan amount: ${loan_amount:,}\n"
            "Evaluate credit risk and provide your assessment."
        )
        response, in_tok, out_tok = self._invoke_bedrock(SYSTEM_PROMPT, user_msg)

        return {
            "agent": self.agent_name,
            "evaluation": response,
            "loan_amount": loan_amount,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
