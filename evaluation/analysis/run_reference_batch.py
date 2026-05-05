#!/usr/bin/env python3
"""
run_reference_batch.py — Automated batch runner for all three reference scenarios
n8n Agentic Pack | Bachelor Thesis Evaluation

Runs SC-01, SC-02, SC-03 (LangChain/LangGraph reference) N times each and saves
results as SC-XX_reference_run002.json, run003.json, etc.
Existing run001.json files are never overwritten.

Usage:
    python evaluation/analysis/run_reference_batch.py --runs 5
    python evaluation/analysis/run_reference_batch.py --runs 5 --scenario SC-01
    python evaluation/analysis/run_reference_batch.py --runs 3 --start-run 6

Environment:
    OPENAI_API_KEY — required (set via $env:OPENAI_API_KEY or export OPENAI_API_KEY)

Scientific context:
    Hevner et al. (2004) require rigorous evaluation of DSR artifacts.
    Multiple runs allow reporting of mean ± std for quantitative metrics,
    replacing the single-run point estimates from the initial pilot runs.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Shared imports (fail fast if missing)
# ---------------------------------------------------------------------------

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langgraph.prebuilt import create_react_agent as lg_react_agent
except ImportError as exc:
    raise SystemExit(
        "[ERROR] Missing dependencies.\n"
        "Run: pip install langchain langchain-openai langgraph langchain-chroma chromadb\n"
        f"Detail: {exc}"
    ) from exc


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    return key


def _token_and_steps(result: dict) -> tuple[int, int]:
    total_tokens = 0
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None) or {}
            total_tokens += usage.get("total_tokens", 0) or 0
    tool_steps = sum(1 for m in result.get("messages", []) if isinstance(m, ToolMessage))
    return total_tokens, tool_steps


def _last_ai_content(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _save(benchmark_result: dict, run_number: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sc = benchmark_result["scenario_id"]
    ap = benchmark_result["approach"]
    run_id = f"run{run_number:03d}"
    benchmark_result["run_id"] = run_id
    path = RESULTS_DIR / f"{sc}_{ap}_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)
    return path


# ===========================================================================
# SC-01: Data Pipeline
# ===========================================================================

def run_sc01_once() -> dict[str, Any]:
    api_key = _get_api_key()

    @tool
    def sql_query(query: str) -> str:
        """Execute SQL SELECT against the ERP database. Input: SQL query string."""
        return json.dumps({
            "rows": [
                {"order_id": 1001, "customer": "Acme GmbH", "total": 1200.50, "status": "pending"},
                {"order_id": 1002, "customer": "Beta AG",   "total": 870.00,  "status": "pending"},
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
                {"order_id": r.get("order_id"), "customer_name": r.get("customer"), "total_eur": r.get("total")}
                for r in data
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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    agent = lg_react_agent(llm, [sql_query, transform_data, erp_push])

    task = (
        "Retrieve all pending orders (status='pending') from the orders table, "
        "transform them to the ERP format (fields: order_id, customer_name, total_eur), "
        "and push the result to the ERP system. Confirm success."
    )

    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency_ms = int((time.time() - t0) * 1000)

    total_tokens, tool_steps = _token_and_steps(result)
    final_answer = _last_ai_content(result)
    success = any(kw in final_answer.lower() for kw in ("erp", "pushed", "imported", "success"))

    return {
        "scenario_id": "SC-01",
        "approach": "reference",
        "run_id": "",  # filled by _save
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


# ===========================================================================
# SC-02: RAG Q&A
# ===========================================================================

DOCUMENTS = [
    Document(
        page_content=(
            "Acme GmbH Q2 2024 Revenue Report: Total revenue reached \u20ac2.4 million. "
            "The Software Licenses category contributed 58% of revenue (\u20ac1.39M), "
            "followed by Professional Services at 32% (\u20ac0.77M) and Hardware at 10% (\u20ac0.24M)."
        ),
        metadata={"source": "erp", "quarter": "Q2-2024", "company": "Acme GmbH", "id": "doc-q2-revenue"},
    ),
    Document(
        page_content=(
            "Acme GmbH Q1 2024 Revenue Report: Total revenue was \u20ac1.9 million. "
            "Software Licenses accounted for 55% (\u20ac1.05M)."
        ),
        metadata={"source": "erp", "quarter": "Q1-2024", "company": "Acme GmbH", "id": "doc-q1-revenue"},
    ),
    Document(
        page_content=(
            "Competitor analysis Q2 2024: Beta AG reported \u20ac1.8M total revenue, "
            "primarily from Hardware (45%)."
        ),
        metadata={"source": "market-report", "quarter": "Q2-2024", "id": "doc-competitor"},
    ),
]


def run_sc02_once() -> dict[str, Any]:
    api_key = _get_api_key()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=DOCUMENTS,
        embedding=embeddings,
        collection_name=f"sc02-eval-ref-{int(time.time())}",  # unique per run
    )

    @tool
    def vector_search(query: str) -> str:
        """Search the internal knowledge base for relevant documents. Input: search query string."""
        results = vectorstore.similarity_search_with_relevance_scores(query, k=3)
        return json.dumps([
            {"text": doc.page_content, "metadata": doc.metadata, "score": round(score, 4)}
            for doc, score in results
        ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    agent = lg_react_agent(llm, [vector_search])

    task = (
        "Answer the following question using only the information in the internal knowledge base: "
        "What was the total revenue of Acme GmbH in Q2 2024, and which product category contributed the most?"
    )

    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency_ms = int((time.time() - t0) * 1000)

    total_tokens, tool_steps = _token_and_steps(result)
    final_answer = _last_ai_content(result)
    success = "2.4" in final_answer or "software" in final_answer.lower()

    return {
        "scenario_id": "SC-02",
        "approach": "reference",
        "run_id": "",
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


# ===========================================================================
# SC-03: Error Recovery
# ===========================================================================

def run_sc03_once() -> dict[str, Any]:
    api_key = _get_api_key()

    _calls = [0]  # mutable counter — reset per run

    @tool
    def sql_query(query: str) -> str:
        """Execute SQL against the database. Input: SQL query string. May return a connection error."""
        _calls[0] += 1
        if _calls[0] == 1:
            return json.dumps({
                "__error": True,
                "message": "Database connection timeout: host db.internal:5432 unreachable after 30s",
            })
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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    agent = lg_react_agent(llm, [sql_query, retry_with_fallback, erp_push])

    task = (
        "Fetch the top-5 overdue invoices from the database (status='overdue', sorted by amount DESC) "
        "and push them to the ERP collection system. Handle any database errors gracefully."
    )

    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency_ms = int((time.time() - t0) * 1000)

    total_tokens, tool_steps = _token_and_steps(result)
    final_answer = _last_ai_content(result)

    tool_messages = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
    error_occurred = any("__error" in (m.content or "") for m in tool_messages)
    error_recovery_success = error_occurred and _calls[0] > 1

    success = any(kw in final_answer.lower() for kw in ("overdue", "invoice", "erp", "imported"))

    return {
        "scenario_id": "SC-03",
        "approach": "reference",
        "run_id": "",
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
        "notes": (
            f"sql_query calls: {_calls[0]}. Error injected: {error_occurred}. "
            f"Recovered: {error_recovery_success}. Final Answer: {final_answer[:200]}"
        ),
    }


# ===========================================================================
# Batch runner
# ===========================================================================

SCENARIO_RUNNERS = {
    "SC-01": run_sc01_once,
    "SC-02": run_sc02_once,
    "SC-03": run_sc03_once,
}


def next_run_number(scenario_id: str, approach: str = "reference") -> int:
    """Return the next available run number (highest existing run number + 1)."""
    existing = list(RESULTS_DIR.glob(f"{scenario_id}_{approach}_run*.json"))
    if not existing:
        return 1
    numbers = []
    for p in existing:
        stem = p.stem  # e.g. SC-01_reference_run003
        parts = stem.split("_")
        for part in parts:
            if part.startswith("run") and part[3:].isdigit():
                numbers.append(int(part[3:]))
    return max(numbers, default=0) + 1


def run_batch(
    scenarios: list[str],
    n_runs: int,
    start_run: int | None = None,
    delay_seconds: float = 1.0,
) -> list[dict]:
    """Run each scenario n_runs times, saving results with sequential run IDs."""
    all_results = []

    for scenario_id in scenarios:
        runner = SCENARIO_RUNNERS.get(scenario_id)
        if runner is None:
            print(f"[SKIP] Unknown scenario: {scenario_id}")
            continue

        run_start = start_run if start_run is not None else next_run_number(scenario_id)

        print(f"\n{'='*60}")
        print(f"  {scenario_id} — {n_runs} run(s) starting at run{run_start:03d}")
        print(f"{'='*60}")

        for i in range(n_runs):
            run_number = run_start + i
            run_label = f"run{run_number:03d}"

            # Skip if file already exists
            out_path = RESULTS_DIR / f"{scenario_id}_reference_{run_label}.json"
            if out_path.exists():
                print(f"  [{run_label}] Already exists — skipping: {out_path.name}")
                continue

            print(f"  [{run_label}] Running {scenario_id}...", end=" ", flush=True)
            try:
                result = runner()
                saved_path = _save(result, run_number)
                status = "✓ success" if result["success"] else "✗ failed"
                print(
                    f"{status} | tokens={result['token_count_total']} "
                    f"steps={result['steps_to_completion']} "
                    f"latency={result['latency_ms']}ms"
                )
                print(f"           Saved: {saved_path.name}")
                all_results.append(result)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR — {exc}")

            if i < n_runs - 1:
                time.sleep(delay_seconds)

    return all_results


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for LangChain reference scenarios.\n"
            "Runs each scenario N times and saves results to evaluation/results/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=4,
        help="Number of additional runs per scenario (default: 4, giving n=5 total with run001)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_RUNNERS.keys()),
        help="Run only a specific scenario (default: all three)",
    )
    parser.add_argument(
        "--start-run",
        type=int,
        default=None,
        help="Override starting run number (default: auto-detect from existing files)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between runs to avoid rate limits (default: 1.0)",
    )
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIO_RUNNERS.keys())

    print(f"\nn8n Agentic Pack — Reference Batch Runner")
    print(f"Scenarios: {scenarios}")
    print(f"Runs per scenario: {args.runs}")
    print(f"Results dir: {RESULTS_DIR.resolve()}")

    results = run_batch(
        scenarios=scenarios,
        n_runs=args.runs,
        start_run=args.start_run,
        delay_seconds=args.delay,
    )

    print(f"\n{'='*60}")
    print(f"  Done. {len(results)} new result file(s) saved.")
    print(f"  Run: python evaluation/analysis/compare.py --latex")
    print(f"  to regenerate statistics tables.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
