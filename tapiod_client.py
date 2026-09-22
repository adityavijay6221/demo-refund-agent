"""A minimal TAPIOD-aware LLM client for this agent.

Governance platforms like this one don't magically "see" an agent just
because it's registered in Backstage - a registered-but-never-connected
agent shows up in the registry with zero run history, forever. The actual
connection is the agent's OWN code choosing to route its LLM calls through
TAPIOD's gateway instead of calling a provider (Anthropic, Bedrock, ...)
directly, carrying two headers on every call:

  X-Team-Key   - which team's policies/budget/exemptions apply
  X-Agent-Id   - which registered agent this call should be attributed to

Both come from the environment, not from code, so the exact same agent.py
routes through governance in one deployment and not in another purely by
env var - no code change needed to onboard or offboard TAPIOD.

This mirrors TestKraft's own TestKraftLLMClient (tapiod-testkraft/
tapiod_testkraft/llm_client.py) minus two things that are specific to that
production deployment and out of place in a small demo agent: the direct-
Bedrock fallback on gateway failure (masks a real outage as a silent
success - fine when TAPIOD fronts your entire production fleet and can't
become a new single point of failure, wrong for something meant to show you
plainly when a call was blocked) and the tapiod SDK package itself (that
package lives inside the Governance monorepo, not on PyPI - this agent is a
standalone repo with no access to it, so this talks to the gateway's plain
HTTP API directly instead).
"""
import os
from typing import Optional

import httpx

TAPIOD_ENABLED = os.getenv("TAPIOD_ENABLED", "false").lower() == "true"
TAPIOD_BASE_URL = os.getenv("TAPIOD_BASE_URL", "http://localhost:14001")
TAPIOD_TEAM_KEY = os.getenv("TAPIOD_TEAM_KEY", "")
TAPIOD_AGENT_ID = os.getenv("TAPIOD_AGENT_ID", "")
TAPIOD_MODEL_ALIAS = os.getenv("TAPIOD_MODEL_ALIAS", "heavy-bedrock")


class TapiodPolicyBlocked(RuntimeError):
    """The gateway refused this call on policy grounds (a secret, PII, a
    prompt-injection pattern, or a custom team policy) - a governance
    decision, not an infrastructure failure. Raised instead of returning
    the refusal text as if it were a real model answer, so a caller can't
    accidentally treat "[This request was blocked...]" as the agent's
    actual output."""


def chat(messages: list[dict], max_tokens: int = 1024) -> str:
    """Send a chat-completions-shaped message list through TAPIOD.

    `messages` is the same list your LLM client already builds - system,
    user, assistant (with tool_calls), and tool-result messages. TAPIOD
    inspects the LAST message's role to know which text this turn's policy
    checks should scan (a plain user/assistant turn, vs. a tool result the
    agent's own previous tool call produced) - see the platform's own
    origin-scoped policy exemptions for why that distinction matters.

    Raises TapiodPolicyBlocked on a real governance block, RuntimeError on
    any other gateway failure (unreachable, misconfigured, 5xx). Neither is
    caught here - a demo agent should surface these plainly, not paper over
    them the way a production fallback would.
    """
    if not TAPIOD_ENABLED:
        raise RuntimeError(
            "TAPIOD_ENABLED is not set - this agent has nothing to route its calls through. "
            "Set TAPIOD_ENABLED=true, TAPIOD_TEAM_KEY and TAPIOD_AGENT_ID to connect it."
        )
    if not TAPIOD_TEAM_KEY or not TAPIOD_AGENT_ID:
        raise RuntimeError(
            "TAPIOD_TEAM_KEY and TAPIOD_AGENT_ID must both be set - a team key alone tells "
            "TAPIOD which team's policies apply, but not which registered agent to attribute "
            "the call to."
        )

    resp = httpx.post(
        f"{TAPIOD_BASE_URL}/api/agent/chat/completions",
        json={"model": TAPIOD_MODEL_ALIAS, "messages": messages, "max_tokens": max_tokens},
        headers={
            "Content-Type": "application/json",
            "X-Team-Key": TAPIOD_TEAM_KEY,
            "X-Agent-Id": TAPIOD_AGENT_ID,
        },
        timeout=60.0,
    )
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # A block surfaces as a real HTTP 403 with the refusal in the body (see
    # main.py's _tapiod_policy_blocked / _tapiod_guard_blocked -> 403
    # translation), not as a 200 whose content happens to read like a
    # refusal - checking the status code is what makes this reliable.
    if resp.status_code == 403:
        raise TapiodPolicyBlocked(content or f"Blocked (HTTP {resp.status_code}), no message body")
    resp.raise_for_status()
    return content
