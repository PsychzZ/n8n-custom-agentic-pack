#!/usr/bin/env python3
"""
SC-03 Error Recovery — LangChain Referenz-Implementierung

Setup:
    pip install langchain langchain-openai langgraph

Ausführung:
    $env:OPENAI_API_KEY="sk-..."
    python3.12 evaluation/reference/sc03_langchain.py
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
    from langgraph.prebuilt import create_react_agent as lg_react_agent
except ImportError:
    print("[ERROR] Install: pip install langchain langchain-openai langgraph")
    raise

# ---------------------------------------------------------------------------
# Tool-Definitionen mit Fehler-Injektion
# ---------------------------------------------------------------------------

_sql_query_calls = [0]  # Mutable counter für Closure

@tool
def sql_query(query: str) -> str:
    """Execute SQL against the database. Input: SQL query string. May return a connection error."""
    _sql_query_calls[0] += 1
    if _sql_query_calls[0] == 1:
        # Erster Aufruf: simulierter Timeout-Fehler
        return json.dumps({
            "__error": True,
            "message": "Database connection timeout: host db.internal:5432 unreachable after 30s"
        })
    # Zweiter+ Aufruf: Erfolg
    return json.dumps({"rows": [
        {"invoice_id": 2001, "customer": "Delta GmbH",  "amount_eur": 15000, "due_date": "2024-01-15", "status": "overdue"},
        {"invoice_id": 2002, "customer": "Epsilon AG",   "amount_eur": 8500,  "due_date": "2024-02-01", "status": "overdue"},
        {"invoice_id": 2003, "customer": "Zeta KG",      "amount_eur": 6200,  "due_date": "2024-02-10", "status": "overdue"},
        {"invoice_id": 2004, "customer": "Eta GmbH",     "amount_eur": 4800,  "due_date": "2024-02-20", "status": "overdue"},
        {"invoice_id": 2005, "customer": "Theta AG",     "amount_eur": 3100,  "due_date": "2024-03-01", "status": "overdue"},
    ]})

@tool
def retry_with_fallback(original_action: str, fallback_query: str = "") -> str:
    """Retry last failed database operation with fallback strategy."""
    return json.dumps({"rows": [
        {"invoice_id": 2001, "customer": "Delta GmbH",  "amount_eur": 15000, "due_date": "2024-01-15", "status": "overdue"},
        {"invoice_id": 2002, "customer": "Epsilon AG",   "amount_eur": 8500,  "due_date": "2024-02-01", "status": "overdue"},
        {"invoice_id": 2003, "customer": "Zeta KG",      "amount_eur": 6200,  "due_date": "2024-02-10", "status": "overdue"},
        {"invoice_id": 2004, "customer": "Eta GmbH",     "amount_eur": 4800,  "due_date": "2024-02-20", "status": "overdue"},
        {"invoice_id": 2005, "customer": "Theta AG",     "amount_eur": 3100,  "due_date": "2024-03-01", "status": "overdue"},
    ]})

@tool
def erp_push(payload: str) -> str:
    """Push invoice data to the ERP collection system. Input: JSON string with rows or payload array."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        data = {}
    records = data.get("payload", data.get("rows", data)) if isinstance(data, dict) else data
    count = len(records) if isinstance(records, list) else 5
    return json.dumps({"status": "ok", "recordsImported": count})

# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def run_sc03() -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    # Reset counter für sauberen Run
    _sql_query_calls[0] = 0

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    tools = [sql_query, retry_with_fallback, erp_push]
    agent = lg_react_agent(llm, tools)

    task = (
        "Fetch the top-5 overdue invoices from the database (status='overdue', sorted by amount DESC) "
        "and push them to the ERP collection system. Handle any database errors gracefully."
    )

    start_time = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency_ms = int((time.time() - start_time) * 1000)

    # Token-Zählung
    total_tokens = 0
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None) or {}
            total_tokens += usage.get("total_tokens", 0) or 0

    tool_steps = sum(1 for m in result.get("messages", []) if isinstance(m, ToolMessage))

    # Prüfen ob Fehler aufgetreten und recovered
    tool_messages = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
    error_occurred = any("__error" in (m.content or "") for m in tool_messages)
    error_recovery_success = error_occurred and _sql_query_calls[0] > 1

    final_answer = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    success = "overdue" in final_answer.lower() or "invoice" in final_answer.lower() or "erp" in final_answer.lower()

    benchmark_result = {
        "scenario_id": "SC-03",
        "approach": "reference",
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_count_total": total_tokens,
        "node_count": 1,
        "steps_to_completion": tool_steps,
        "success": success,
        "error_recovery_success": error_recovery_success,
        "latency_ms": latency_ms,
        "transparency_score": 2,
        "controllability_score": 1,
        "integration_score": 3,
        "notes": f"sql_query calls: {_sql_query_calls[0]}. Error injected: {error_occurred}. Recovered: {error_recovery_success}. Final Answer: {final_answer[:200]}",
    }

    out_path = "evaluation/results/SC-03_reference_run001.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BENCHMARK ERGEBNIS: SC-03 Referenz (LangChain Error Recovery)")
    print(f"{'='*60}")
    print(f"Final Answer:     {final_answer[:200]}")
    print(f"Token (gesamt):   {total_tokens}")
    print(f"Tool-Schritte:    {tool_steps}")
    print(f"Latenz:           {latency_ms} ms")
    print(f"Fehler injiziert: {error_occurred}")
    print(f"Recovered:        {error_recovery_success}")
    print(f"Erfolg:           {success}")
    print(f"\nErgebnis gespeichert: {out_path}")
    print(f"{'='*60}")

    return benchmark_result


if __name__ == "__main__":
    run_sc03()
