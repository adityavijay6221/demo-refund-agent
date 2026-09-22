"""
Refund Processing Agent - a small LangGraph agent that reads a customer
refund request, looks up the order and the customer's identity-verification
record, and drafts a refund decision summary for a human agent to approve.

Built as a governance-platform demo: a genuinely small, real agent (not a
fixture) that lives in its own repository outside any org the platform
auto-scans, to be onboarded manually instead of discovered.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

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
    llm = ChatAnthropic(model="claude-sonnet-4-5")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Refund request:\n{state['request_text']}\n\nCustomer record:\n{state['customer_record']}"},
    ]
    response = llm.invoke(messages)
    return {**state, "recommendation": response.content}


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
    print(run(sample_request))
