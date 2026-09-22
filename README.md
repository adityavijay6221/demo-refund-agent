# Refund Processing Agent

A small LangGraph agent for customer support: reads a refund request, looks
up the order and the customer's identity-verification record, and drafts a
recommendation for a human agent to approve. It never issues the refund
itself.

## How it works

1. `lookup_node` pulls the order id off the request and calls
   `lookup_customer_record` (a support-tooling lookup) for the order and
   identity-verification details.
2. `recommend_node` hands that result to the model - via TAPIOD, when
   configured - as a real tool-call turn (an assistant message carrying the
   call, a tool message carrying what it returned), and asks for the final
   refund recommendation.

## Governance

This agent routes its LLM calls through [TAPIOD](tapiod_client.py) rather
than calling a model provider directly. Whether a given run is governed is
entirely an environment decision, not a code one:

```bash
export TAPIOD_ENABLED=true
export TAPIOD_BASE_URL=http://localhost:14001   # your TAPIOD gateway
export TAPIOD_TEAM_KEY=tgk_...                  # from your team's Governance tab
export TAPIOD_AGENT_ID=refund-processing-agent  # this agent's registered slug
```

With those set, every run is attributed to this agent and this team in the
governance platform - real cost, real policy checks, real blocks if content
trips a check the team has adopted, real exemptions if one's been approved
for this agent.

## Run it

```bash
pip install -r requirements.txt
python agent.py
```
