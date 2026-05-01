"""
market_summarizer_agent.py
Agent: MarketSummarizer
Tool:  summarize_market_conditions
Domain: Macro indicators — inflation, yields, sentiment, sector rotation
"""

from typing import Any, Dict, Optional
from .base_agent import BaseFinancialAgent, AgentConfig


AGENT_CFG = AgentConfig(
    agent_id="market-summarizer-v1",
    description="Summarizes market conditions and macro economic indicators",
    version="1.0.0",
    tool_name="summarize_market_conditions",
    tags={"domain": "finance", "type": "summarization"},
)

SYSTEM_PROMPT = """You are a macroeconomic research analyst.
Based on the market condition snapshot provided, write a concise market summary
covering: market sentiment, key macro indicators (inflation, yields, GDP),
sector rotation signals, and an overall market outlook (BULLISH/NEUTRAL/BEARISH).
Keep the summary under 250 words."""


class MarketSummarizerAgent(BaseFinancialAgent):
    """Summarizes market conditions from macro indicator snapshot."""

    def __init__(self):
        super().__init__(AGENT_CFG)

    # ------------------------------------------------------------------ #
    #  Tool: summarize_market_conditions                                  #
    # ------------------------------------------------------------------ #

    def run_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derives market condition labels from raw macro data.
        Inputs (all optional with sensible defaults):
          - cpi_yoy_pct: float       (CPI year-over-year %)
          - fed_funds_rate: float    (current Fed Funds Rate %)
          - ten_yr_yield: float      (10-yr Treasury yield %)
          - sp500_ytd_pct: float     (S&P 500 YTD return %)
          - vix: float               (VIX index level)
          - gdp_growth_pct: float    (annualized GDP growth %)
          - leading_sectors: list    (outperforming sectors)
          - lagging_sectors: list    (underperforming sectors)
        """
        cpi = input_data.get("cpi_yoy_pct", 3.0)
        ffr = input_data.get("fed_funds_rate", 5.25)
        yield_10yr = input_data.get("ten_yr_yield", 4.5)
        sp500_ytd = input_data.get("sp500_ytd_pct", 8.0)
        vix = input_data.get("vix", 18.0)
        gdp = input_data.get("gdp_growth_pct", 2.1)
        leading = input_data.get("leading_sectors", ["Technology", "Healthcare"])
        lagging = input_data.get("lagging_sectors", ["Utilities", "Real Estate"])

        # Derived labels
        inflation_env = "HIGH" if cpi > 4 else ("MODERATE" if cpi > 2.5 else "LOW")
        rate_env = "RESTRICTIVE" if ffr > yield_10yr else "ACCOMMODATIVE"
        vix_signal = "FEARFUL" if vix > 30 else ("CAUTIOUS" if vix > 20 else "CALM")
        equity_trend = "STRONG" if sp500_ytd > 10 else ("WEAK" if sp500_ytd < 0 else "MODERATE")

        # Overall outlook heuristic
        bullish_signals = sum([
            cpi < 3.5,
            gdp > 2.0,
            vix < 20,
            sp500_ytd > 5,
        ])
        outlook = "BULLISH" if bullish_signals >= 3 else ("BEARISH" if bullish_signals <= 1 else "NEUTRAL")

        return {
            "cpi_yoy_pct": cpi,
            "inflation_environment": inflation_env,
            "fed_funds_rate": ffr,
            "rate_environment": rate_env,
            "ten_yr_yield": yield_10yr,
            "sp500_ytd_pct": sp500_ytd,
            "equity_trend": equity_trend,
            "vix": vix,
            "vix_signal": vix_signal,
            "gdp_growth_pct": gdp,
            "leading_sectors": leading,
            "lagging_sectors": lagging,
            "overall_outlook": outlook,
        }

    # ------------------------------------------------------------------ #
    #  AgentCore entry point                                               #
    # ------------------------------------------------------------------ #

    def run(self, user_query: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        market_snapshot = self.run_tool(context)
        user_message = (
            f"User question: {user_query}\n\n"
            f"Market conditions snapshot:\n{market_snapshot}\n\n"
            "Write your market summary."
        )
        return self.invoke_llm(SYSTEM_PROMPT, user_message)
