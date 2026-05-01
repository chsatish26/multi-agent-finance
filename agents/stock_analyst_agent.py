"""Stock analyst agent — analyzes a ticker and produces a recommendation."""
from .base_agent import BaseFinanceAgent

SYSTEM_PROMPT = """You are an expert stock analyst. Given a ticker symbol and optional context,
produce a structured analysis with: summary, key metrics to watch, sentiment (bullish/bearish/neutral),
and a recommendation (buy/hold/sell). Be concise and factual."""


class StockAnalystAgent(BaseFinanceAgent):
    def __init__(self):
        super().__init__("StockAnalyst")

    def _run(self, input_data: dict) -> dict:
        ticker = input_data.get("ticker", "UNKNOWN")
        context = input_data.get("context", "No additional context provided.")

        user_msg = f"Analyze {ticker}. Additional context: {context}"
        response, in_tok, out_tok = self._invoke_bedrock(SYSTEM_PROMPT, user_msg)

        return {
            "agent": self.agent_name,
            "ticker": ticker,
            "analysis": response,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
