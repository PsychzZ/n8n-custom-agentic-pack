#!/usr/bin/env python3
"""
record_n8n_run.py — Interactive helper to record manual n8n benchmark runs
n8n Agentic Pack | Bachelor Thesis Evaluation

After running a workflow in n8n, use this script to log the result
in the correct JSON format without manually editing files.

Usage:
    python evaluation/analysis/record_n8n_run.py

The script prompts for:
  - Scenario (SC-01 / SC-02 / SC-03)
  - Approach (artefact / baseline)
  - Token count (from n8n log or OpenAI Dashboard)
  - Steps to completion (from AgentState history length)
  - Latency in ms (from n8n execution time)
  - Success (y/n)
  - Error recovery (y/n/skip for non-SC-03)
  - Notes (optional)

Qualitative scores (transparency, controllability, integration) are fixed
per approach since they reflect structural architecture properties, not
run-to-run variance. This is noted in the thesis methodology.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Fixed qualitative scores per approach (structural, not performance-variable)
QUALITATIVE_SCORES = {
    "artefact":  {"transparency_score": 5, "controllability_score": 4, "integration_score": 4},
    "baseline":  {"transparency_score": 2, "controllability_score": 2, "integration_score": 3},
}

SCENARIOS = ["SC-01", "SC-02", "SC-03"]
APPROACHES = ["artefact", "baseline"]


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val if val else default


def prompt_int(label: str, default: int = 0) -> int:
    while True:
        val = prompt(label, str(default))
        try:
            return int(val)
        except ValueError:
            print(f"    Please enter an integer.")


def prompt_bool(label: str) -> bool | None:
    while True:
        val = input(f"  {label} (y/n/skip): ").strip().lower()
        if val in ("y", "yes", "true", "1"):
            return True
        if val in ("n", "no", "false", "0"):
            return False
        if val in ("skip", "s", ""):
            return None
        print("    Enter y, n, or skip.")


def next_run_number(scenario_id: str, approach: str) -> int:
    existing = list(RESULTS_DIR.glob(f"{scenario_id}_{approach}_run*.json"))
    numbers = []
    for p in existing:
        for part in p.stem.split("_"):
            if part.startswith("run") and part[3:].isdigit():
                numbers.append(int(part[3:]))
    return max(numbers, default=0) + 1


def main() -> None:
    print("\nn8n Agentic Pack — Manual Run Recorder")
    print("="*45)

    # Scenario
    print(f"\n  Scenarios: {', '.join(SCENARIOS)}")
    scenario_id = ""
    while scenario_id not in SCENARIOS:
        scenario_id = input("  Scenario: ").strip().upper()
        if scenario_id not in SCENARIOS:
            print(f"    Choose from {SCENARIOS}")

    # Approach
    print(f"\n  Approaches: {', '.join(APPROACHES)}")
    approach = ""
    while approach not in APPROACHES:
        approach = input("  Approach: ").strip().lower()
        if approach not in APPROACHES:
            print(f"    Choose from {APPROACHES}")

    # Auto-detect run number
    run_number = next_run_number(scenario_id, approach)
    print(f"\n  Auto-detected next run number: run{run_number:03d}")
    override = input("  Press Enter to accept, or type a different number: ").strip()
    if override.isdigit():
        run_number = int(override)

    print(f"\n  Entering results for {scenario_id} / {approach} / run{run_number:03d}")
    print("  (Press Enter to use default values)")

    token_count_total = prompt_int("Token count total (from OpenAI Dashboard)", 0)
    steps_to_completion = prompt_int("Steps to completion (ReAct iterations / tool calls)", 0)
    latency_ms = prompt_int("Latency in ms (from n8n execution time)", 0)

    print()
    success_raw = input("  Success? (y/n): ").strip().lower()
    success = success_raw in ("y", "yes", "true", "1")

    error_recovery_success = None
    if scenario_id == "SC-03":
        print()
        error_recovery_success = prompt_bool("Error recovery successful?")

    notes = prompt("Notes (optional)", "")

    qualitative = QUALITATIVE_SCORES[approach]

    benchmark_result = {
        "scenario_id": scenario_id,
        "approach": approach,
        "run_id": f"run{run_number:03d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_count_total": token_count_total,
        "node_count": 6 if approach == "artefact" else (4 if scenario_id == "SC-02" else 5),
        "steps_to_completion": steps_to_completion,
        "success": success,
        "error_recovery_success": error_recovery_success,
        "latency_ms": latency_ms,
        **qualitative,
        "notes": notes,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_label = f"run{run_number:03d}"
    out_path = RESULTS_DIR / f"{scenario_id}_{approach}_{run_label}.json"

    if out_path.exists():
        overwrite = input(f"\n  File already exists: {out_path.name}\n  Overwrite? (y/N): ").strip().lower()
        if overwrite not in ("y", "yes"):
            print("  Aborted.")
            return

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    print(f"\n  Saved: {out_path.name}")
    print(f"  Run: python evaluation/analysis/compare.py --latex  to regenerate tables.")
    print()


if __name__ == "__main__":
    main()
