"""
Agent registry — discovers and routes requests to the correct finance agent.
"""
from agents import StockAnalystAgent, MarketSummarizerAgent, CreditEvaluatorAgent, RiskAssessorAgent


class AgentRegistry:
    def __init__(self):
        self._agents = {
            "stock_analyst": StockAnalystAgent,
            "market_summarizer": MarketSummarizerAgent,
            "credit_evaluator": CreditEvaluatorAgent,
            "risk_assessor": RiskAssessorAgent,
        }
        self._instances: dict = {}

    def get(self, agent_name: str):
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self._agents)}")
        if agent_name not in self._instances:
            self._instances[agent_name] = self._agents[agent_name]()
        return self._instances[agent_name]

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    def invoke(self, agent_name: str, input_data: dict) -> dict:
        return self.get(agent_name).invoke(input_data)
