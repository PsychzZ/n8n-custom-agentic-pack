#!/usr/bin/env python3
"""
compare.py — Benchmark Analysis Script
n8n Agentic Pack | Bachelor Thesis Evaluation

Reads raw result JSON files from evaluation/results/ and produces:
  1. A summary statistics table (mean ± std per metric per approach)
  2. Bar charts comparing the three approaches per scenario
  3. A LaTeX-ready table for the thesis

Usage:
    python evaluation/analysis/compare.py
    python evaluation/analysis/compare.py --scenario SC-01
    python evaluation/analysis/compare.py --output-dir docs/thesis-text/figures/

References:
    Hevner et al. (2004) — Design Science evaluation requires rigorous comparison
    of the artifact against a baseline and a reference implementation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependencies — only needed for plotting
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("[WARN] pandas / matplotlib not installed. Run: pip install pandas matplotlib numpy")
    print("[INFO] Falling back to text-only output.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis" / "output"

QUANTITATIVE_METRICS = [
    "token_count_total",
    "node_count",
    "steps_to_completion",
    "success",
    "latency_ms",
]

QUALITATIVE_METRICS = [
    "transparency_score",
    "controllability_score",
    "integration_score",
]

APPROACHES = ["baseline", "artefact", "reference"]
SCENARIOS = ["SC-01", "SC-02", "SC-03"]

METRIC_LABELS = {
    "token_count_total": "Total Tokens",
    "node_count": "Node Count",
    "steps_to_completion": "Steps to Completion",
    "success": "Success Rate",
    "error_recovery_success": "Error Recovery Rate",
    "latency_ms": "Latency (ms)",
    "transparency_score": "Transparency (1–5)",
    "controllability_score": "Controllability (1–5)",
    "integration_score": "Integration Ease (1–5)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """
    Load all result JSON files from the results directory.

    File naming convention: {scenario_id}_{approach}_{run_id}.json
    Example: SC-01_artefact_run001.json

    Returns a flat list of result dicts, each augmented with scenario/approach/run_id keys.
    """
    results = []
    if not results_dir.exists():
        print(f"[WARN] Results directory not found: {results_dir}")
        return results

    for path in sorted(results_dir.glob("*.json")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            print(f"[SKIP] Unexpected filename format: {path.name}")
            continue
        scenario_id = parts[0]
        approach = parts[1]
        run_id = "_".join(parts[2:])

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["scenario_id"] = scenario_id
            data["approach"] = approach
            data["run_id"] = run_id
            results.append(data)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Could not parse {path.name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------


def compute_summary(
    results: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compute mean and std for each metric per (scenario, approach) combination.

    Returns nested dict: summary[scenario_id][approach][metric] = {"mean": x, "std": y, "n": z}
    """
    from collections import defaultdict

    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for r in results:
        sc = r.get("scenario_id", "unknown")
        ap = r.get("approach", "unknown")
        for metric in metrics:
            if metric in r:
                buckets[sc][ap][metric].append(float(r[metric]))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for sc, approaches in buckets.items():
        summary[sc] = {}
        for ap, metrics_data in approaches.items():
            summary[sc][ap] = {}
            for metric, values in metrics_data.items():
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
                std = variance ** 0.5
                summary[sc][ap][metric] = {"mean": mean, "std": std, "n": len(values)}

    return summary


def print_summary_table(
    summary: dict[str, dict[str, dict[str, Any]]],
    metrics: list[str],
) -> None:
    """Print a human-readable summary table to stdout."""
    all_scenarios = sorted(summary.keys())
    for sc in all_scenarios:
        print(f"\n{'='*70}")
        print(f"  Scenario: {sc}")
        print(f"{'='*70}")
        header = f"{'Metric':<30}" + "".join(f"{ap:<20}" for ap in APPROACHES)
        print(header)
        print("-" * 70)
        for metric in metrics:
            label = METRIC_LABELS.get(metric, metric)
            row = f"{label:<30}"
            for ap in APPROACHES:
                stats = summary.get(sc, {}).get(ap, {}).get(metric)
                if stats:
                    row += f"{stats['mean']:.2f} ± {stats['std']:.2f}   "
                else:
                    row += f"{'N/A':<20}"
            print(row)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------


def generate_latex_table(
    summary: dict[str, dict[str, dict[str, Any]]],
    scenario_id: str,
    metrics: list[str],
) -> str:
    """Generate a LaTeX tabular for a single scenario comparison."""
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Benchmark Results — {scenario_id}}}",
        f"  \\label{{tab:benchmark-{scenario_id.lower()}}}",
        "  \\begin{tabular}{l" + "c" * len(APPROACHES) + "}",
        "    \\toprule",
        "    Metric & " + " & ".join(f"\\textbf{{{ap.capitalize()}}}" for ap in APPROACHES) + " \\\\",
        "    \\midrule",
    ]

    sc_data = summary.get(scenario_id, {})
    for metric in metrics:
        label = METRIC_LABELS.get(metric, metric)
        cells = []
        for ap in APPROACHES:
            stats = sc_data.get(ap, {}).get(metric)
            if stats:
                cells.append(f"${stats['mean']:.2f} \\pm {stats['std']:.2f}$")
            else:
                cells.append("N/A")
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    summary: dict[str, dict[str, dict[str, Any]]],
    metrics: list[str],
    output_dir: Path,
) -> None:
    """Generate bar charts comparing the three approaches per scenario."""
    if not HAS_PLOTTING:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#95a5a6", "artefact": "#2ecc71", "reference": "#3498db"}
    x = np.arange(len(SCENARIOS))
    width = 0.25

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        label = METRIC_LABELS.get(metric, metric)

        for idx, ap in enumerate(APPROACHES):
            means = []
            stds = []
            for sc in SCENARIOS:
                stats = summary.get(sc, {}).get(ap, {}).get(metric)
                if stats:
                    means.append(stats["mean"])
                    stds.append(stats["std"])
                else:
                    means.append(0.0)
                    stds.append(0.0)

            ax.bar(
                x + idx * width,
                means,
                width,
                yerr=stds,
                label=ap.capitalize(),
                color=colors.get(ap, "#888"),
                capsize=4,
                alpha=0.85,
            )

        ax.set_xlabel("Scenario")
        ax.set_ylabel(label)
        ax.set_title(f"Benchmark Comparison — {label}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(SCENARIOS)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        filename = output_dir / f"benchmark_{metric}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Saved: {filename}")


# ---------------------------------------------------------------------------
# Result file template generator
# ---------------------------------------------------------------------------


def create_result_template(
    scenario_id: str = "SC-01",
    approach: str = "artefact",
    run_id: str = "run001",
) -> dict[str, Any]:
    """Create an empty result JSON template for manual filling."""
    return {
        "scenario_id": scenario_id,
        "approach": approach,
        "run_id": run_id,
        "timestamp": "",
        "token_count_total": 0,
        "node_count": 0,
        "steps_to_completion": 0,
        "success": False,
        "error_recovery_success": None,
        "latency_ms": 0,
        "transparency_score": 0,
        "controllability_score": 0,
        "integration_score": 0,
        "notes": "",
    }


def scaffold_results(results_dir: Path = RESULTS_DIR) -> None:
    """Create empty result templates for all scenario × approach combinations."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        for ap in APPROACHES:
            filename = results_dir / f"{sc}_{ap}_run001.json"
            if not filename.exists():
                template = create_result_template(sc, ap, "run001")
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(template, f, indent=2)
                print(f"[SCAFFOLD] Created: {filename.name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark analysis for the n8n Agentic Pack thesis evaluation."
    )
    parser.add_argument("--scenario", help="Filter to a single scenario (e.g. SC-01)")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for generated plots and LaTeX files",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Create empty result template files",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX tables to stdout",
    )
    args = parser.parse_args()

    if args.scaffold:
        scaffold_results()
        return

    results = load_results()
    if not results:
        print("[INFO] No result files found. Run with --scaffold to create templates.")
        sys.exit(0)

    all_metrics = QUANTITATIVE_METRICS + QUALITATIVE_METRICS
    summary = compute_summary(results, all_metrics)

    if args.scenario:
        filtered_summary = {k: v for k, v in summary.items() if k == args.scenario}
    else:
        filtered_summary = summary

    print_summary_table(filtered_summary, all_metrics)

    if args.latex:
        for sc in filtered_summary:
            print(f"\n% --- LaTeX table for {sc} ---")
            print(generate_latex_table(filtered_summary, sc, all_metrics))

    if HAS_PLOTTING:
        output_dir = Path(args.output_dir)
        plot_comparison(filtered_summary, all_metrics, output_dir)


if __name__ == "__main__":
    main()
