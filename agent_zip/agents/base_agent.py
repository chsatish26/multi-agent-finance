# Paste base_agent.py content here
"""
base_agent.py
Base class for all financial agents.
Handles LLM initialization from YAML config and AgentCore registration.
"""

import os
import json
import logging
import yaml
import boto3
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    agent_id: str
    description: str
    version: str
    tool_name: str
    tags: Dict[str, str] = field(default_factory=dict)


class BaseFinancialAgent(ABC):
    """
    Base class for all financial agents.
    - Loads LLM config from YAML
    - Provides Bedrock client
    - Handles AgentCore registration metadata
    """

    CONFIG_PATH = os.environ.get(
        "AGENT_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
    )

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.config = self._load_config()
        self.llm_config = self.config["llm"]
        self._bedrock_client = None  # lazy — created on first LLM call
        logger.info(f"[{self.agent_config.agent_id}] Initialized (model={self.llm_config['model_id']})")

    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #

    def _load_config(self) -> Dict[str, Any]:
        with open(self.CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    @property
    def bedrock_client(self):
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.llm_config["region"],
            )
        return self._bedrock_client

    # ------------------------------------------------------------------ #
    #  LLM call                                                            #
    # ------------------------------------------------------------------ #

    def invoke_llm(self, system_prompt: str, user_message: str) -> str:
        """Call Claude via Bedrock with the configured model."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.llm_config["max_tokens"],
            "temperature": self.llm_config["temperature"],
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.llm_config["model_id"],
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    # ------------------------------------------------------------------ #
    #  AgentCore metadata (used by registry)                              #
    # ------------------------------------------------------------------ #

    def get_registry_metadata(self) -> Dict[str, Any]:
        return {
            "agentId": self.agent_config.agent_id,
            "description": self.agent_config.description,
            "version": self.agent_config.version,
            "tool": self.agent_config.tool_name,
            "tags": self.agent_config.tags,
            "status": "ACTIVE",
        }

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Each agent implements its single domain tool here."""

    @abstractmethod
    def run(self, user_query: str, context: Optional[Dict] = None) -> str:
        """Entry point called by AgentCore runtime."""