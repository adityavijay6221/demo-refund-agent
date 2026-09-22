# Refund Processing Agent

A small LangGraph agent for customer support: reads a refund request, looks
up the order and the customer's identity-verification record, and drafts a
recommendation for a human agent to approve. It never issues the refund
itself.

## How it works

1. `lookup_node` pulls the order id off the request and calls
   `lookup_customer_record` (a support-tooling lookup) for the order and
   identity-verification details.
2. `recommend_node` asks Claude to draft the refund recommendation from the
   request plus the customer record.

## Run it

```bash
pip install -r requirements.txt
python agent.py
```
