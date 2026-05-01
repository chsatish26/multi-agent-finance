"""
runtime_app.py
AgentCore Runtime entrypoint using BedrockAgentCoreApp.
Receives JSON payload: {"agent_id": "...", "query": "...", "context": {...}}
"""

import os
import logging

os.environ.setdefault(
    "AGENT_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "config", "config.yaml"),
)

from bedrock_agentcore import BedrockAgentCoreApp
from agents import (
    StockAnalystAgent,
    RiskAssessorAgent,
    MarketSummarizerAgent,
    CreditEvaluatorAgent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("runtime_app")

app = BedrockAgentCoreApp()

_AGENTS: dict | None = None

def _get_agents() -> dict:
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = {
            "stock-analyst-v1":     StockAnalystAgent(),
            "risk-assessor-v1":     RiskAssessorAgent(),
            "market-summarizer-v1": MarketSummarizerAgent(),
            "credit-evaluator-v1":  CreditEvaluatorAgent(),
        }
        logger.info(f"Loaded agents: {list(_AGENTS.keys())}")
    return _AGENTS


@app.entrypoint
def invoke(payload):
    agent_id = (payload.get("agent_id") or "").strip()
    query    = (payload.get("query")    or "").strip()
    context  = payload.get("context", {}) or {}

    AGENTS = _get_agents()

    if not agent_id:
        return {"error": "Missing 'agent_id'", "available_agents": list(AGENTS.keys())}
    if not query:
        return {"error": "Missing 'query'"}

    agent = AGENTS.get(agent_id)
    if not agent:
        return {"error": f"Unknown agent: '{agent_id}'", "available_agents": list(AGENTS.keys())}

    logger.info(f"Invoking {agent_id} | query={query[:80]}")

    try:
        tool_output  = agent.run_tool(context)
        llm_response = agent.run(query, context)
        return {
            "agent_id":    agent_id,
            "query":       query,
            "tool_output": tool_output,
            "response":    llm_response,
        }
    except Exception as e:
        logger.exception(f"Agent {agent_id} failed")
        return {"error": str(e), "agent_id": agent_id}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
