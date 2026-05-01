"""
stock_analyst_agent.py
Agent: StockAnalyst
Tool:  calculate_financial_ratios
Domain: Equity analysis — P/E, EPS, ROE, Debt-to-Equity
"""

from typing import Any, Dict, Optional
from .base_agent import BaseFinancialAgent, AgentConfig


AGENT_CFG = AgentConfig(
    agent_id="stock-analyst-v1",
    description="Analyzes stock price trends and financial ratios",
    version="1.0.0",
    tool_name="calculate_financial_ratios",
    tags={"domain": "finance", "type": "analysis"},
)

SYSTEM_PROMPT = """You are a financial analyst specializing in equity research.
You have access to calculated financial ratios. Use them to provide a concise,
factual analysis. Focus on: P/E valuation, profitability (ROE, EPS), and
leverage (Debt-to-Equity). Keep responses professional and under 200 words."""


class StockAnalystAgent(BaseFinancialAgent):
    """Analyzes stocks using financial ratios (single internal tool)."""

    def __init__(self):
        super().__init__(AGENT_CFG)

    # ------------------------------------------------------------------ #
    #  Tool: calculate_financial_ratios                                   #
    # ------------------------------------------------------------------ #

    def run_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates key financial ratios from raw financial data.
        No external API calls — pure computation.
        """
        price = input_data.get("stock_price", 0)
        eps = input_data.get("eps", 1)
        net_income = input_data.get("net_income", 0)
        equity = input_data.get("shareholders_equity", 1)
        total_debt = input_data.get("total_debt", 0)

        return {
            "pe_ratio": round(price / eps, 2) if eps else None,
            "roe_percent": round((net_income / equity) * 100, 2) if equity else None,
            "debt_to_equity": round(total_debt / equity, 2) if equity else None,
            "eps": eps,
            "stock_price": price,
            "ticker": input_data.get("ticker", "N/A"),
        }

    # ------------------------------------------------------------------ #
    #  AgentCore entry point                                               #
    # ------------------------------------------------------------------ #

    def run(self, user_query: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        ratios = self.run_tool(context)
        user_message = (
            f"User question: {user_query}\n\n"
            f"Financial ratios computed:\n{ratios}\n\n"
            "Provide your analysis."
        )
        return self.invoke_llm(SYSTEM_PROMPT, user_message)
