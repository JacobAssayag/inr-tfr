"""
summarize_results.py
====================
Scan the results/ directory and produce a summary report.

Outputs
-------
    <output_dir>/summary.csv       all metrics from all runs
    <output_dir>/best_runs.csv     best run per condition (lowest rel_err_t)
    <output_dir>/comparison.png    P(t,f) comparison: marginal vs moments vs ROI
    <output_dir>/metrics_table.tex LaTeX table for thesis insertion

Usage
-----
    python summarize_results.py [--results-dir results] [--output-dir reports]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
METRIC_COLS = [
    "rel_err_t",
    "rel_err_f",
    "parseval_err",
    "symmetry_err",
    "mass",
    "seed",
    "signal",
    "num_epochs",
]


def _load_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from a run directory; return None if missing."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        data = json.load(f)
    data["run_dir"] = str(run_dir)
    data["condition"] = _infer_condition(run_dir.name)
    return data


def _infer_condition(folder_name: str) -> str:
    """Infer condition label from timestamped folder name."""
    # Format: <timestamp>_<signal>_<condition>_seed<n>
    parts = folder_name.split("_")
    # Drop timestamp (first two parts: YYYYMMDD, HHMMSS)
    return "_".join(parts[2:]) if len(parts) > 2 else folder_name


def _nan_if_missing(d: dict, key: str) -> float:
    val = d.get(key, float("nan"))
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def scan_results(results_dir: Path) -> list[dict]:
    """Recursively scan results_dir for run folders containing metrics.json."""
    rows = []
    if not results_dir.exists():
        print(f"WARNING: results directory not found: {results_dir}")
        return rows
    for candidate in sorted(results_dir.iterdir()):
        if candidate.is_dir():
            m = _load_metrics(candidate)
            if m is not None:
                rows.append(m)
    return rows


def build_summary_df(rows: list[dict]) -> list[dict]:
    """Flatten rows into a list of dicts with canonical metric columns."""
    output = []
    cols = ["run_dir", "condition"] + METRIC_COLS
    for r in rows:
        flat = {c: _nan_if_missing(r, c) for c in cols}
        flat["run_dir"] = r.get("run_dir", "")
        flat["condition"] = r.get("condition", "")
        flat["signal"] = r.get("signal", "")
        output.append(flat)
    return output


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        print(f"WARNING: no rows to write to {path}")
        return
    import csv
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def find_best_runs(summary: list[dict]) -> list[dict]:
    """
    For each unique condition, find the run with the lowest rel_err_t.
    Ties are broken by rel_err_f.
    """
    from collections import defaultdict
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in summary:
        by_condition[row["condition"]].append(row)

    best = []
    for cond, runs in by_condition.items():
        valid = [r for r in runs if not math.isnan(r.get("rel_err_t", float("nan")))]
        if not valid:
            continue
        best_run = min(valid, key=lambda r: (r["rel_err_t"], r.get("rel_err_f", 1e9)))
        best.append(best_run)

    best.sort(key=lambda r: r["condition"])
    return best


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def make_bar_comparison(summary: list[dict], output_path: Path) -> None:
    """
    Bar chart comparing rel_err_t and symmetry_err across conditions.
    """
    # Aggregate per condition: median over seeds
    from collections import defaultdict
    import statistics

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in summary:
        by_cond[r["condition"]].append(r)

    conditions = sorted(by_cond.keys())
    rel_err_t_vals = []
    sym_err_vals = []

    for cond in conditions:
        runs = by_cond[cond]
        rets = [r["rel_err_t"] for r in runs if not math.isnan(r["rel_err_t"])]
        syms = [r["symmetry_err"] for r in runs
                if not math.isnan(r.get("symmetry_err", float("nan")))]
        rel_err_t_vals.append(statistics.median(rets) if rets else float("nan"))
        sym_err_vals.append(statistics.median(syms) if syms else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(conditions))
    width = 0.6

    ax = axes[0]
    bars = ax.bar(x, rel_err_t_vals, width, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("rel_err_t (median)")
    ax.set_title("Time marginal relative error by condition")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, sym_err_vals, width, color="firebrick", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("symmetry_err (median)")
    ax.set_title("Symmetry error by condition")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved → {output_path}")


# --------------------------------------------------------------------------- #
# LaTeX table                                                                  #
# --------------------------------------------------------------------------- #
def _fmt(v: float, precision: int = 4) -> str:
    """Format a float for LaTeX; replace NaN with '---'."""
    if math.isnan(v):
        return r"\text{---}"
    return f"{v:.{precision}f}"


def make_latex_table(best_runs: list[dict], output_path: Path) -> None:
    """
    Generate a LaTeX table of best-run metrics per condition.

    Columns: Condition | Signal | Seed | rel\\_err\\_t | rel\\_err\\_f |
             symmetry\\_err | parseval\\_err
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{INR Time-Frequency Experiment Results (best run per condition)}",
        r"\label{tab:inr_tfr_results}",
        r"\begin{tabular}{llr rrrr}",
        r"\toprule",
        (
            r"Condition & Signal & Seed "
            r"& $\varepsilon_t$ & $\varepsilon_f$ "
            r"& $\varepsilon_{\mathrm{sym}}$ & $\varepsilon_{\mathrm{Parseval}}$ \\"
        ),
        r"\midrule",
    ]

    for r in best_runs:
        cond = r["condition"].replace("_", r"\_")
        sig = str(r.get("signal", "")).replace("_", r"\_")
        seed = str(int(r.get("seed", -1)))
        rel_t = _fmt(_nan_if_missing(r, "rel_err_t"))
        rel_f = _fmt(_nan_if_missing(r, "rel_err_f"))
        sym = _fmt(_nan_if_missing(r, "symmetry_err"), precision=2)
        pars = _fmt(_nan_if_missing(r, "parseval_err"), precision=2)
        lines.append(
            f"{cond} & {sig} & {seed} & {rel_t} & {rel_f} & {sym} & {pars} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved → {output_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise results from the results/ directory."
    )
    parser.add_argument(
        "--results-dir", "-r",
        default="results",
        help="Root directory containing timestamped run folders.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="reports",
        help="Directory to write summary outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {results_dir}")
    rows = scan_results(results_dir)
    print(f"Found {len(rows)} run(s).")

    if not rows:
        print("No results found. Run some experiments first.", file=sys.stderr)
        sys.exit(0)

    summary = build_summary_df(rows)
    write_csv(summary, output_dir / "summary.csv")

    best_runs = find_best_runs(summary)
    write_csv(best_runs, output_dir / "best_runs.csv")

    make_bar_comparison(summary, output_dir / "comparison.png")
    make_latex_table(best_runs, output_dir / "metrics_table.tex")

    print(f"\nAll outputs written to: {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
