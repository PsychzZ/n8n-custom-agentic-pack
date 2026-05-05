#!/usr/bin/env python3
"""
SC-02 RAG Q&A — LangChain Referenz-Implementierung

Setup:
    pip install langchain langchain-openai langgraph langchain-chroma chromadb

Ausführung:
    $env:OPENAI_API_KEY="sk-..."
    python3.12 evaluation/reference/sc02_langchain.py
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
except ImportError:
    print("[ERROR] Install: pip install langchain langchain-openai langgraph langchain-chroma chromadb")
    raise

# ---------------------------------------------------------------------------
# Dokumente (identisch mit SC-02-Szenario)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    Document(
        page_content="Acme GmbH Q2 2024 Revenue Report: Total revenue reached \u20ac2.4 million. "
                     "The Software Licenses category contributed 58% of revenue (\u20ac1.39M), "
                     "followed by Professional Services at 32% (\u20ac0.77M) and Hardware at 10% (\u20ac0.24M).",
        metadata={"source": "erp", "quarter": "Q2-2024", "company": "Acme GmbH", "id": "doc-q2-revenue"}
    ),
    Document(
        page_content="Acme GmbH Q1 2024 Revenue Report: Total revenue was \u20ac1.9 million. "
                     "Software Licenses accounted for 55% (\u20ac1.05M).",
        metadata={"source": "erp", "quarter": "Q1-2024", "company": "Acme GmbH", "id": "doc-q1-revenue"}
    ),
    Document(
        page_content="Competitor analysis Q2 2024: Beta AG reported \u20ac1.8M total revenue, "
                     "primarily from Hardware (45%).",
        metadata={"source": "market-report", "quarter": "Q2-2024", "id": "doc-competitor"}
    ),
]

# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def run_sc02() -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")

    # In-memory ChromaDB mit OpenAI Embeddings (konsistent mit Artefakt)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = Chroma.from_documents(
        documents=DOCUMENTS,
        embedding=embeddings,
        collection_name="sc02-eval-ref",
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
    from langgraph.prebuilt import create_react_agent as lg_react_agent
    agent = lg_react_agent(llm, [vector_search])

    task = (
        "Answer the following question using only the information in the internal knowledge base: "
        "What was the total revenue of Acme GmbH in Q2 2024, and which product category contributed the most?"
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

    final_answer = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    success = "2.4" in final_answer or "software" in final_answer.lower()

    benchmark_result = {
        "scenario_id": "SC-02",
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

    out_path = "evaluation/results/SC-02_reference_run001.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BENCHMARK ERGEBNIS: SC-02 Referenz (LangChain RAG)")
    print(f"{'='*60}")
    print(f"Final Answer:   {final_answer[:200]}")
    print(f"Token (gesamt): {total_tokens}")
    print(f"Tool-Schritte:  {tool_steps}")
    print(f"Latenz:         {latency_ms} ms")
    print(f"Erfolg:         {success}")
    print(f"\nErgebnis gespeichert: {out_path}")
    print(f"{'='*60}")

    return benchmark_result


if __name__ == "__main__":
    run_sc02()
