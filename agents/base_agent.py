"""
Base agent with CloudWatch metric emission for AgentCore observability dashboard.
"""
import json
import time
import logging
from datetime import datetime, timezone

import boto3
import yaml

logger = logging.getLogger(__name__)

_config_cache = None


def _load_config():
    global _config_cache
    if _config_cache is None:
        config_path = __file__.replace("agents/base_agent.py", "config/config.yaml")
        with open(config_path) as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


class BaseFinanceAgent:
    """
    Base class for all finance agents.
    Handles Bedrock invocation, CloudWatch metrics, and structured logging.
    """

    def __init__(self, agent_name: str):
        cfg = _load_config()
        self.agent_name = agent_name
        self.region = cfg["aws"]["region"]
        self.model_id = cfg["bedrock"]["model_id"]
        self.max_tokens = cfg["bedrock"]["max_tokens"]
        self.namespace = cfg["observability"]["metrics_namespace"]
        self.log_group = cfg["observability"]["log_group"]

        self._bedrock = boto3.client("bedrock-runtime", region_name=self.region)
        self._cw = boto3.client("cloudwatch", region_name=self.region)
        self._logs = boto3.client("logs", region_name=self.region)
        self._ensure_log_stream()

    # ── CloudWatch helpers ─────────────────────────────────────────

    def _ensure_log_stream(self):
        stream = f"{self.agent_name}/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}"
        try:
            self._logs.create_log_stream(logGroupName=self.log_group, logStreamName=stream)
        except self._logs.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            logger.warning("Could not create log stream: %s", e)
        self._log_stream = stream

    def _emit(self, metric_name: str, value: float, unit: str = "Count", extra_dims: list = None):
        dims = [{"Name": "AgentName", "Value": self.agent_name}]
        if extra_dims:
            dims.extend(extra_dims)
        try:
            self._cw.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{"MetricName": metric_name, "Value": value, "Unit": unit, "Dimensions": dims}],
            )
        except Exception as e:
            logger.warning("Metric emit failed (%s): %s", metric_name, e)

    def _log_event(self, level: str, message: str, extra: dict = None):
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "level": level,
                   "agent": self.agent_name, "message": message}
        if extra:
            payload.update(extra)
        try:
            self._logs.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self._log_stream,
                logEvents=[{"timestamp": int(time.time() * 1000), "message": json.dumps(payload)}],
            )
        except Exception as e:
            logger.warning("Log event failed: %s", e)

    # ── Bedrock invocation ─────────────────────────────────────────

    def _invoke_bedrock(self, system_prompt: str, user_message: str) -> tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        resp = self._bedrock.invoke_model(modelId=self.model_id, body=json.dumps(body))
        data = json.loads(resp["body"].read())
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    # ── Public invoke ──────────────────────────────────────────────

    def invoke(self, input_data: dict) -> dict:
        """Invoke the agent and emit CloudWatch metrics. Subclasses override _run()."""
        start = time.time()
        self._emit("RuntimeInvocations", 1)
        self._log_event("INFO", "Invocation started", {"input_keys": list(input_data.keys())})

        try:
            result = self._run(input_data)
            elapsed_ms = (time.time() - start) * 1000

            self._emit("RuntimeLatency", elapsed_ms, unit="Milliseconds")
            if "input_tokens" in result:
                self._emit("InputTokens", result["input_tokens"])
                self._emit("OutputTokens", result.get("output_tokens", 0))

            self._log_event("INFO", "Invocation completed", {"elapsed_ms": round(elapsed_ms, 1)})
            return result

        except Exception as e:
            self._emit("RuntimeErrors", 1)
            self._log_event("ERROR", str(e), {"error_type": type(e).__name__})
            raise

    def _run(self, input_data: dict) -> dict:
        raise NotImplementedError
