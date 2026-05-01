"""
agentcore_registry.py
Handles registration of all financial agents with AWS AgentCore Registry.

Remote mode uses the real `bedrock-agentcore-control` boto3 client (v1.42.87+).
Local mode uses an in-memory store for offline/test use.

Prerequisites (remote mode):
  1. Create a registry in the AgentCore console (one-time).
  2. Set env vars:
       AGENTCORE_MODE=remote
       AGENTCORE_REGISTRY_ID=<your-registry-id>   # extracted from registry ARN
       AGENTCORE_REGION=us-east-1                  # or your region
  3. SageMaker execution role needs:
       bedrock-agentcore:CreateRegistryRecord
       bedrock-agentcore:ListRegistryRecords
       bedrock-agentcore:GetRegistryRecord
       bedrock-agentcore:DeleteRegistryRecord
       bedrock-agentcore:SubmitRegistryRecordForApproval
"""

import os
import json
import logging
import boto3
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── env config ────────────────────────────────────────────────────────────────
AGENTCORE_MODE        = os.environ.get("AGENTCORE_MODE", "local")
AGENTCORE_REGISTRY_ID = os.environ.get("AGENTCORE_REGISTRY_ID", "")
AGENTCORE_REGION      = os.environ.get("AGENTCORE_REGION", "us-east-1")


class AgentCoreRegistry:
    """
    Remote mode  → real AWS bedrock-agentcore-control API calls
    Local mode   → in-memory store (no AWS required; used for tests / demo)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config       = config
        self._mode        = AGENTCORE_MODE
        self._registry_id = AGENTCORE_REGISTRY_ID
        self._region      = AGENTCORE_REGION
        self._local_store: Dict[str, Dict] = {}

        if self._mode == "remote":
            if not self._registry_id:
                raise ValueError(
                    "AGENTCORE_REGISTRY_ID env var is required in remote mode.\n"
                    "Get it from the AgentCore console → Registry → your registry ARN\n"
                    "  ARN format: arn:aws:bedrock-agentcore:<region>:<account>:registry/<registry-id>\n"
                    "  Set: export AGENTCORE_REGISTRY_ID=<registry-id>"
                )
            self._control = boto3.client("bedrock-agentcore-control", region_name=self._region)
            logger.info(f"AgentCoreRegistry → remote (registry={self._registry_id}, region={self._region})")
        else:
            self._control = None
            logger.info("AgentCoreRegistry → local (in-memory)")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def register(self, agent_metadata: Dict[str, Any]) -> bool:
        return self._remote_register(agent_metadata) if self._mode == "remote" \
            else self._local_register(agent_metadata)

    def register_all(self, agents: list) -> Dict[str, bool]:
        return {
            a.agent_config.agent_id: self.register(a.get_registry_metadata())
            for a in agents
        }

    def deregister(self, agent_id: str) -> bool:
        return self._remote_deregister(agent_id) if self._mode == "remote" \
            else self._local_deregister(agent_id)

    def heartbeat(self, agent_id: str) -> bool:
        if self._mode == "remote":
            logger.info(f"[REMOTE] Heartbeat not applicable — use AgentCore runtime health checks.")
            return True
        return self._local_heartbeat(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        return self._remote_list() if self._mode == "remote" else list(self._local_store.values())

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self._remote_get(agent_id) if self._mode == "remote" \
            else self._local_store.get(agent_id)

    def print_registry(self):
        agents = self.list_agents()
        print(f"\n{'='*60}")
        print(f"  AgentCore Registry — {len(agents)} agent(s) registered")
        print(f"{'='*60}")
        for a in agents:
            print(f"  ► {a.get('agentId') or a.get('name', 'unknown')} [{a.get('status', 'UNKNOWN')}]")
            print(f"    version   : {a.get('version')}")
            print(f"    tool      : {a.get('tool')}")
            print(f"    tags      : {a.get('tags')}")
            reg = a.get('registeredAt') or a.get('createdAt', '')
            print(f"    registered: {reg}")
            if self._mode == "remote":
                print(f"    recordId  : {a.get('recordId', '')}")
                print(f"    recordArn : {a.get('recordArn', '')}")
            print()

    # ------------------------------------------------------------------ #
    #  Remote — real AWS bedrock-agentcore-control calls                  #
    # ------------------------------------------------------------------ #

    def _build_descriptor(self, meta: Dict[str, Any]) -> Dict:
        content = {
            "agentId":     meta["agentId"],
            "description": meta["description"],
            "version":     meta["version"],
            "tool":        meta["tool"],
            "tags":        meta.get("tags", {}),
            "status":      meta.get("status", "ACTIVE"),
        }
        return {
            "custom": {
                "inlineContent": json.dumps(content)
            }
        }

    def _remote_register(self, meta: Dict[str, Any]) -> bool:
        agent_id = meta["agentId"]
        try:
            resp = self._control.create_registry_record(
                registryId=self._registry_id,
                name=agent_id,
                descriptorType="CUSTOM",
                descriptors=self._build_descriptor(meta),
            )
            record_id  = resp.get("recordId", "")
            record_arn = resp.get("recordArn", "")
            status     = resp.get("status", "")

            self._local_store[agent_id] = {
                **meta,
                "recordId":     record_id,
                "recordArn":    record_arn,
                "status":       status,
                "registeredAt": datetime.now(timezone.utc).isoformat(),
            }
            logger.info(f"[REMOTE] Registered {agent_id} → recordId={record_id} status={status}")

            # Submit for approval if not auto-approved
            if record_id and status not in ("APPROVED", "ACTIVE"):
                self._submit_for_approval(record_id)

            return True

        except Exception as e:
            if "ConflictException" in type(e).__name__ or "already exists" in str(e).lower():
                logger.warning(f"[REMOTE] {agent_id} already registered — skipping.")
                return True
            logger.error(f"[REMOTE] Failed to register {agent_id}: {e}")
            return False

    def _submit_for_approval(self, record_id: str):
        try:
            resp = self._control.submit_registry_record_for_approval(
                registryId=self._registry_id,
                recordId=record_id,
            )
            logger.info(f"[REMOTE] Submitted {record_id} for approval → {resp.get('status')}")
        except Exception as e:
            logger.warning(f"[REMOTE] Could not submit {record_id} for approval: {e}")

    def _remote_deregister(self, agent_id: str) -> bool:
        record = self._local_store.get(agent_id)
        record_id = record.get("recordId") if record else None
        if not record_id:
            logger.warning(f"[REMOTE] No recordId cached for {agent_id} — cannot deregister.")
            return False
        try:
            self._control.delete_registry_record(
                registryId=self._registry_id,
                recordId=record_id,
            )
            self._local_store.pop(agent_id, None)
            logger.info(f"[REMOTE] Deregistered {agent_id}")
            return True
        except Exception as e:
            logger.error(f"[REMOTE] Failed to deregister {agent_id}: {e}")
            return False

    def _remote_list(self) -> List[Dict[str, Any]]:
        """Returns cache populated during register_all. Call _remote_list_fresh() for live API."""
        return list(self._local_store.values())

    def _remote_list_fresh(self) -> List[Dict[str, Any]]:
        try:
            resp = self._control.list_registry_records(registryId=self._registry_id)
            return resp.get("registryRecords", [])
        except Exception as e:
            logger.error(f"[REMOTE] Failed to list registry records: {e}")
            return []

    def _remote_get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self._local_store.get(agent_id)

    # ------------------------------------------------------------------ #
    #  Local (in-memory)                                                   #
    # ------------------------------------------------------------------ #

    def _local_register(self, meta: Dict[str, Any]) -> bool:
        agent_id = meta["agentId"]
        self._local_store[agent_id] = {
            **meta,
            "registeredAt":  datetime.now(timezone.utc).isoformat(),
            "lastHeartbeat": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"[LOCAL] Registered agent: {agent_id}")
        return True

    def _local_deregister(self, agent_id: str) -> bool:
        removed = self._local_store.pop(agent_id, None)
        if removed:
            logger.info(f"[LOCAL] Deregistered: {agent_id}")
        else:
            logger.warning(f"[LOCAL] Not found: {agent_id}")
        return removed is not None

    def _local_heartbeat(self, agent_id: str) -> bool:
        if agent_id in self._local_store:
            self._local_store[agent_id]["lastHeartbeat"] = datetime.now(timezone.utc).isoformat()
            return True
        return False