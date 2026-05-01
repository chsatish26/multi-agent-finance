"""
Multi-Agent Finance — entry point.
Runs all four agents with sample inputs to populate the CloudWatch dashboard.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from registry import AgentRegistry


def run_demo():
    registry = AgentRegistry()

    tasks = [
        ("stock_analyst", {"ticker": "NVDA", "context": "Q1 2026 earnings beat estimates by 12%."}),
        ("market_summarizer", {
            "indicators": ["S&P 500 +0.8%", "10Y yield 4.3%", "VIX 18.2"],
            "headlines": ["Fed holds rates steady", "Jobs report beats expectations"],
        }),
        ("credit_evaluator", {
            "applicant": {"credit_score": 720, "annual_income": 95000, "debt_to_income": 0.32,
                          "employment_years": 5},
            "loan_amount": 250000,
        }),
        ("risk_assessor", {
            "portfolio": [
                {"ticker": "AAPL", "weight": 0.25, "sector": "Technology"},
                {"ticker": "MSFT", "weight": 0.20, "sector": "Technology"},
                {"ticker": "JPM",  "weight": 0.15, "sector": "Financials"},
                {"ticker": "BRK.B","weight": 0.10, "sector": "Financials"},
                {"ticker": "JNJ",  "weight": 0.10, "sector": "Healthcare"},
                {"ticker": "AGG",  "weight": 0.20, "sector": "Fixed Income"},
            ],
            "total_value": 500000,
        }),
    ]

    results = {}
    for agent_name, input_data in tasks:
        print(f"\n{'='*60}")
        print(f"  Running: {agent_name}")
        print(f"{'='*60}")
        try:
            result = registry.invoke(agent_name, input_data)
            results[agent_name] = result
            # Print the main text output
            text_key = next((k for k in ("analysis", "summary", "evaluation", "risk_report") if k in result), None)
            if text_key:
                print(result[text_key])
            print(f"\n  Tokens — in: {result.get('input_tokens', 0)}, out: {result.get('output_tokens', 0)}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[agent_name] = {"error": str(e)}

    print(f"\n{'='*60}")
    print("  All agents complete. Metrics emitted to CloudWatch.")
    print(f"  Dashboard: https://us-east-1.console.aws.amazon.com/cloudwatch/home"
          f"?region=us-east-1#dashboards:name=AgentCore-Monitoring")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    run_demo()
