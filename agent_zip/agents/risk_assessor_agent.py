"""
risk_assessor_agent.py
Agent: RiskAssessor
Tool:  assess_portfolio_risk
Domain: Portfolio risk — VaR, Sharpe Ratio, Beta, Volatility
"""

import math
from typing import Any, Dict, List, Optional
from .base_agent import BaseFinancialAgent, AgentConfig


AGENT_CFG = AgentConfig(
    agent_id="risk-assessor-v1",
    description="Evaluates portfolio risk and Value-at-Risk metrics",
    version="1.0.0",
    tool_name="assess_portfolio_risk",
    tags={"domain": "finance", "type": "risk"},
)

SYSTEM_PROMPT = """You are a risk management specialist.
You analyze portfolio risk metrics including VaR, Sharpe ratio, Beta, and volatility.
Provide a clear, concise risk assessment and recommend whether the portfolio risk
level is LOW, MEDIUM, or HIGH. Be precise and data-driven."""


class RiskAssessorAgent(BaseFinancialAgent):
    """Assesses portfolio risk metrics (single internal tool)."""

    def __init__(self):
        super().__init__(AGENT_CFG)

    # ------------------------------------------------------------------ #
    #  Tool: assess_portfolio_risk                                        #
    # ------------------------------------------------------------------ #

    def run_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes risk metrics from portfolio returns data.
        Inputs:
          - returns: List[float]  (daily return percentages)
          - portfolio_value: float
          - benchmark_returns: List[float] (market returns, same length)
          - risk_free_rate: float (annualized, e.g. 0.05 for 5%)
        """
        returns: List[float] = input_data.get("returns", [])
        portfolio_value: float = input_data.get("portfolio_value", 100_000)
        benchmark: List[float] = input_data.get("benchmark_returns", [])
        rf_daily: float = input_data.get("risk_free_rate", 0.05) / 252

        if not returns:
            return {"error": "No returns data provided"}

        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_dev = math.sqrt(variance)

        # VaR at 95% confidence (parametric)
        z_95 = 1.645
        var_95 = portfolio_value * z_95 * std_dev

        # Sharpe ratio (annualized)
        excess_return = mean_r - rf_daily
        sharpe = (excess_return / std_dev) * math.sqrt(252) if std_dev else 0

        # Beta vs benchmark
        beta = None
        if benchmark and len(benchmark) == n:
            bm_mean = sum(benchmark) / n
            cov = sum((returns[i] - mean_r) * (benchmark[i] - bm_mean) for i in range(n)) / (n - 1)
            bm_var = sum((b - bm_mean) ** 2 for b in benchmark) / (n - 1)
            beta = round(cov / bm_var, 3) if bm_var else None

        # Risk level classification
        if std_dev < 0.01 and sharpe > 1.5:
            risk_level = "LOW"
        elif std_dev > 0.025 or sharpe < 0.5:
            risk_level = "HIGH"
        else:
            risk_level = "MEDIUM"

        return {
            "portfolio_value": portfolio_value,
            "daily_volatility_pct": round(std_dev * 100, 4),
            "annualized_volatility_pct": round(std_dev * math.sqrt(252) * 100, 2),
            "var_95_usd": round(var_95, 2),
            "sharpe_ratio": round(sharpe, 3),
            "beta": beta,
            "risk_level": risk_level,
            "observations": n,
        }

    # ------------------------------------------------------------------ #
    #  AgentCore entry point                                               #
    # ------------------------------------------------------------------ #

    def run(self, user_query: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        risk_metrics = self.run_tool(context)
        user_message = (
            f"User question: {user_query}\n\n"
            f"Computed risk metrics:\n{risk_metrics}\n\n"
            "Provide your risk assessment."
        )
        return self.invoke_llm(SYSTEM_PROMPT, user_message)
