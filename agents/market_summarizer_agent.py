"""Market summarizer agent — summarizes macro market conditions."""
from .base_agent import BaseFinanceAgent

SYSTEM_PROMPT = """You are a market analyst. Given a list of market indicators or headlines,
produce a concise market summary covering: overall sentiment, key drivers, sectors to watch,
and a short-term outlook. Keep it under 300 words."""


class MarketSummarizerAgent(BaseFinanceAgent):
    def __init__(self):
        super().__init__("MarketSummarizer")

    def _run(self, input_data: dict) -> dict:
        indicators = input_data.get("indicators", [])
        headlines = input_data.get("headlines", [])

        user_msg = (
            f"Market indicators: {', '.join(indicators) if indicators else 'not provided'}.\n"
            f"Headlines: {'; '.join(headlines) if headlines else 'not provided'}.\n"
            "Provide a market summary."
        )
        response, in_tok, out_tok = self._invoke_bedrock(SYSTEM_PROMPT, user_msg)

        return {
            "agent": self.agent_name,
            "summary": response,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
