#!/usr/bin/env python3
"""
SC-01 Data Pipeline — LangChain ReAct Agent (Referenz-Implementierung)

Führt dasselbe Szenario wie SC-01 mit LangChain + LangGraph durch,
um Vergleichsdaten (token_count, steps_to_completion, latency_ms)
für die Thesis-Evaluation zu erheben.

Setup:
    pip install langchain langchain-openai langgraph

Ausführung:
    $env:OPENAI_API_KEY="sk-..."   # PowerShell
    python3.12 evaluation/reference/sc01_langchain.py
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
except ImportError:
    print("[ERROR] Install dependencies: pip install langchain langchain-openai langgraph")
    raise

# ---------------------------------------------------------------------------
# Tool-Definitionen (simulierte Antworten — identisch mit SC-01-Szenario)
# ---------------------------------------------------------------------------

@tool
def sql_query(query: str) -> str:
    """Execute SQL SELECT against the ERP database. Input: SQL query string."""
    return json.dumps({
        "rows": [
            {"order_id": 1001, "customer": "Acme GmbH", "total": 1200.50, "status": "pending"},
            {"order_id": 1002, "customer": "Beta AG",   "total": 870.00,  "status": "pending"}
        ]
    })

@tool
def transform_data(rows: str) -> str:
    """Transform raw SQL rows to ERP format. Input: JSON string with rows array."""
    try:
        data = json.loads(rows)
    except (json.JSONDecodeError, TypeError):
        data = {}
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        data = []
    return json.dumps({
        "payload": [
            {
                "order_id": row.get("order_id"),
                "customer_name": row.get("customer"),
                "total_eur": row.get("total")
            }
            for row in data
        ]
    })

@tool
def erp_push(payload: str) -> str:
    """Push transformed payload to the ERP system. Input: JSON string with payload array."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        data = {}
    records = data.get("payload", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        records = []
    return json.dumps({"status": "ok", "recordsImported": len(records)})

# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def run_sc01() -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    tools = [sql_query, transform_data, erp_push]
    agent = create_react_agent(llm, tools)

    task = (
        "Retrieve all pending orders (status='pending') from the orders table, "
        "transform them to the ERP format (fields: order_id, customer_name, total_eur), "
        "and push the result to the ERP system. Confirm success."
    )

    start_time = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency_ms = int((time.time() - start_time) * 1000)

    # Token-Zählung aus usage_metadata der Messages
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    steps = 0

    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            steps += 1
            usage = getattr(msg, "usage_metadata", None) or {}
            total_tokens      += usage.get("total_tokens",      0) or 0
            prompt_tokens     += usage.get("input_tokens",      0) or 0
            completion_tokens += usage.get("output_tokens",     0) or 0

    # Letztes AIMessage = Final Answer
    final_answer = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    success = (
        "erp" in final_answer.lower()
        or "pushed" in final_answer.lower()
        or "imported" in final_answer.lower()
        or "success" in final_answer.lower()
    )

    # Schritte = Anzahl Tool-Calls (ToolMessages)
    tool_steps = sum(1 for m in result.get("messages", []) if isinstance(m, ToolMessage))

    benchmark_result = {
        "scenario_id": "SC-01",
        "approach": "reference",
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_count_total": total_tokens,
        "node_count": 1,
        "steps_to_completion": tool_steps,
        "success": success,
        "error_recovery_success": None,
        "latency_ms": latency_ms,
        "transparency_score": 2,
        "controllability_score": 1,
        "integration_score": 3,
        "notes": f"Final Answer: {final_answer[:300]}",
    }

    out_path = "evaluation/results/SC-01_reference_run001.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BENCHMARK ERGEBNIS: SC-01 Referenz (LangChain + LangGraph)")
    print(f"{'='*60}")
    print(f"Final Answer:     {final_answer[:200]}")
    print(f"Token (gesamt):   {total_tokens}")
    print(f"Tool-Schritte:    {tool_steps}")
    print(f"Latenz:           {latency_ms} ms")
    print(f"Erfolg:           {success}")
    print(f"\nErgebnis gespeichert: {out_path}")
    print(f"{'='*60}")

    return benchmark_result


if __name__ == "__main__":
    run_sc01()
