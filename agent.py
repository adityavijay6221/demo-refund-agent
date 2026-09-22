"""
Refund Processing Agent - a small LangGraph agent that reads a customer
refund request, looks up the order and the customer's identity-verification
record, and drafts a refund decision summary for a human agent to approve.

Routes its LLM calls through TAPIOD (see tapiod_client.py) when
TAPIOD_ENABLED is set - the environment decides whether this agent's calls
are governed, not a code change. Run `python agent.py` directly to see a
real run against your own TAPIOD gateway, complete with real Backstage
attribution (agents.last_active_at, ai_sessions, blocked_call_log,
agent_policy_exemption_applications) - not a simulated call from a console.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

import tapiod_client

SYSTEM_PROMPT = """You are a customer-support refund assistant. Given a
refund request, look up the order and the customer's identity-verification
record, then draft a one-paragraph refund recommendation for a human agent
to approve. You never issue the refund yourself - you only recommend."""


class RefundState(TypedDict):
    request_text: str
    customer_record: str
    recommendation: str


@tool
def lookup_customer_record(order_id: str) -> str:
    """Look up a customer's order and identity-verification record by order id."""
    # Demo fixture: the same shape a real support-tooling lookup would
    # return, including the identity-verification field on file.
    return (
        f"Order: {order_id}\n"
        "Customer: Priya Anand\n"
        "Verified via: SSN on file 521-74-8843\n"
        "Order total: $89.00 - delivered 2026-09-10, return window open"
    )


@tool
def flag_for_review(reason: str) -> str:
    """Flag this refund request for manual review instead of auto-drafting a recommendation."""
    return f"Flagged for manual review: {reason}"


def lookup_node(state: RefundState) -> RefundState:
    order_id = state["request_text"].split("Order:")[-1].split("\n")[0].strip()
    record = lookup_customer_record.invoke({"order_id": order_id})
    return {**state, "customer_record": record}


def recommend_node(state: RefundState) -> RefundState:
    # The lookup above already ran as a real tool call (lookup_node, just
    # above) - this turn hands its result back to the model exactly the way
    # a genuine agent loop would: an assistant message carrying the tool
    # call, followed by a tool-role message carrying what it returned. That
    # shape is what tells TAPIOD this text is a TOOL RESULT rather than
    # something the user typed - the axis agent_policy_exemptions scopes
    # exemptions by (see the platform's own origin-scoping design).
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["request_text"]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_customer_record", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": state["customer_record"]},
    ]
    recommendation = tapiod_client.chat(messages)
    return {**state, "recommendation": recommendation}


def build_graph():
    graph = StateGraph(RefundState)
    graph.add_node("lookup", lookup_node)
    graph.add_node("recommend", recommend_node)
    graph.set_entry_point("lookup")
    graph.add_edge("lookup", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def run(request_text: str) -> str:
    app = build_graph()
    result = app.invoke({"request_text": request_text, "customer_record": "", "recommendation": ""})
    return result["recommendation"]


if __name__ == "__main__":
    sample_request = "Order: ORD-88213\nReason: item arrived damaged\nRequested refund: $89.00"
    try:
        print(run(sample_request))
    except tapiod_client.TapiodPolicyBlocked as e:
        print(f"Blocked by policy: {e}")
