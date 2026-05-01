"""
credit_evaluator_agent.py
Agent: CreditEvaluator
Tool:  evaluate_credit_profile
Domain: Credit scoring — DTI, LTV, credit score bands, loan eligibility
"""

from typing import Any, Dict, Optional
from .base_agent import BaseFinancialAgent, AgentConfig


AGENT_CFG = AgentConfig(
    agent_id="credit-evaluator-v1",
    description="Evaluates credit scores and loan eligibility",
    version="1.0.0",
    tool_name="evaluate_credit_profile",
    tags={"domain": "finance", "type": "credit"},
)

SYSTEM_PROMPT = """You are a credit analyst at a financial institution.
Based on the credit profile evaluation provided, give a professional assessment
covering: creditworthiness, key risk factors, and a recommendation of
APPROVE, CONDITIONAL APPROVE, or DECLINE for the loan application.
Be objective and cite specific metrics from the data."""


class CreditEvaluatorAgent(BaseFinancialAgent):
    """Evaluates credit profiles and loan eligibility (single internal tool)."""

    def __init__(self):
        super().__init__(AGENT_CFG)

    # ------------------------------------------------------------------ #
    #  Tool: evaluate_credit_profile                                      #
    # ------------------------------------------------------------------ #

    def run_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a borrower's credit profile.
        Inputs:
          - credit_score: int        (300-850)
          - annual_income: float     (USD)
          - monthly_debt: float      (existing monthly obligations)
          - loan_amount: float       (requested loan)
          - loan_term_months: int    (e.g. 360 for 30yr)
          - interest_rate: float     (annual, e.g. 0.065)
          - property_value: float    (for LTV, 0 if personal loan)
          - employment_years: float  (years at current employer)
        """
        score = input_data.get("credit_score", 650)
        income = input_data.get("annual_income", 80_000)
        monthly_debt = input_data.get("monthly_debt", 500)
        loan_amount = input_data.get("loan_amount", 300_000)
        term = input_data.get("loan_term_months", 360)
        rate = input_data.get("interest_rate", 0.065)
        prop_value = input_data.get("property_value", 0)
        emp_years = input_data.get("employment_years", 3.0)

        monthly_income = income / 12

        # Monthly mortgage/loan payment (amortization)
        r = rate / 12
        if r > 0 and term > 0:
            monthly_payment = loan_amount * (r * (1 + r) ** term) / ((1 + r) ** term - 1)
        else:
            monthly_payment = loan_amount / term if term else 0

        total_monthly_debt = monthly_debt + monthly_payment
        dti = (total_monthly_debt / monthly_income) * 100 if monthly_income else 999
        ltv = (loan_amount / prop_value) * 100 if prop_value else None

        # Credit score band
        if score >= 750:
            score_band = "EXCELLENT"
        elif score >= 700:
            score_band = "GOOD"
        elif score >= 650:
            score_band = "FAIR"
        else:
            score_band = "POOR"

        # Eligibility heuristics
        flags = []
        if dti > 43:
            flags.append("HIGH_DTI")
        if score < 620:
            flags.append("LOW_CREDIT_SCORE")
        if ltv and ltv > 95:
            flags.append("HIGH_LTV")
        if emp_years < 2:
            flags.append("SHORT_EMPLOYMENT_HISTORY")

        if not flags:
            recommendation = "APPROVE"
        elif len(flags) == 1 and "HIGH_DTI" not in flags and "LOW_CREDIT_SCORE" not in flags:
            recommendation = "CONDITIONAL_APPROVE"
        else:
            recommendation = "DECLINE"

        return {
            "credit_score": score,
            "score_band": score_band,
            "dti_pct": round(dti, 2),
            "monthly_payment_usd": round(monthly_payment, 2),
            "ltv_pct": round(ltv, 2) if ltv else "N/A (personal loan)",
            "employment_years": emp_years,
            "risk_flags": flags,
            "preliminary_recommendation": recommendation,
            "loan_amount": loan_amount,
            "annual_income": income,
        }

    # ------------------------------------------------------------------ #
    #  AgentCore entry point                                               #
    # ------------------------------------------------------------------ #

    def run(self, user_query: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        credit_profile = self.run_tool(context)
        user_message = (
            f"User question: {user_query}\n\n"
            f"Credit profile evaluation:\n{credit_profile}\n\n"
            "Provide your credit assessment and recommendation."
        )
        return self.invoke_llm(SYSTEM_PROMPT, user_message)
