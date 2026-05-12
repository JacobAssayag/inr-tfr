"""
analyze_results.py
==================
Aggregates all INR-TFR training run outputs into a single CSV summary and
generates comparison plots grouped by loss type.

Usage
-----
    python analyze_results.py --results_dir results

The script expects a directory tree such as::

    results/
        run_001/
            config.yaml
            metrics.json
            loss_curve.csv
            final_figure.png
        run_002/
            ...

Output
------
    results/summary.csv
    results/analysis_plots/*.png
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering – no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names that are considered "expected"; used for documentation only.
EXPECTED_COLUMNS = [
    "signal_type",
    "model_type",
    "loss_type",
    "seed",
    "lr",
    "lambda_moment",
    "lambda_roi",
    "lambda_sparsity",
    "lambda_tv",
    "time_rel_error",
    "freq_rel_error",
    "symmetry_error",
    "ridge_error",
    "spread",
    "entropy",
    "ambiguity_score",
    "peak_localization_error",
    "final_loss",
    "runtime_seconds",
    "figure_path",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flatten(obj, parent_key="", sep="."):
    """Recursively flatten a nested dict/list into a single-level dict.

    Parameters
    ----------
    obj : dict | list | scalar
        Object to flatten.
    parent_key : str
        Key prefix accumulated during recursion.
    sep : str
        Separator inserted between nested key levels.

    Returns
    -------
    dict
        Flat key→value mapping.
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items


def _load_json(path):
    """Load a JSON file; return empty dict on failure with a warning."""
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return {}


def _load_yaml(path):
    """Load a YAML file; return empty dict on failure with a warning."""
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Run discovery and aggregation
# ---------------------------------------------------------------------------

def find_run_dirs(results_dir):
    """Walk *results_dir* and return every directory that contains metrics.json.

    Parameters
    ----------
    results_dir : Path
        Root directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of run directories.
    """
    run_dirs = []
    for root, _dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            run_dirs.append(Path(root))
    return sorted(run_dirs)


def build_run_record(run_dir):
    """Build a flat dict representing one run.

    Parameters
    ----------
    run_dir : Path
        Directory for this run.

    Returns
    -------
    dict
        Row suitable for a pandas DataFrame.  Missing values are np.nan.
    bool
        True if the run is considered *incomplete* (metrics.json missing or
        config.yaml missing).
    """
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    figure_path = run_dir / "final_figure.png"

    incomplete = False
    record = {}

    # --- Metadata -----------------------------------------------------------
    record["run_id"] = run_dir.name
    record["run_path"] = str(run_dir)
    record["metrics_path"] = str(metrics_path) if metrics_path.exists() else np.nan
    record["config_path"] = str(config_path) if config_path.exists() else np.nan
    record["figure_path"] = str(figure_path) if figure_path.exists() else np.nan

    # --- Config -------------------------------------------------------------
    if config_path.exists():
        cfg = _load_yaml(config_path)
        if cfg:
            record.update(_flatten(cfg))
        else:
            incomplete = True
            warnings.warn(f"[INCOMPLETE] Empty or unreadable config.yaml in {run_dir}")
    else:
        incomplete = True
        warnings.warn(f"[INCOMPLETE] config.yaml not found in {run_dir}")

    # --- Metrics ------------------------------------------------------------
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        if metrics:
            record.update(_flatten(metrics))
        else:
            incomplete = True
            warnings.warn(f"[INCOMPLETE] Empty or unreadable metrics.json in {run_dir}")
    else:
        # This should not happen because we filtered by metrics.json existence,
        # but guard anyway.
        incomplete = True
        warnings.warn(f"[INCOMPLETE] metrics.json not found in {run_dir}")

    return record, incomplete


def collect_runs(results_dir):
    """Discover all runs and build a summary DataFrame.

    Parameters
    ----------
    results_dir : Path
        Root directory to scan.

    Returns
    -------
    pd.DataFrame
        One row per run.  Missing fields are NaN.
    int
        Number of incomplete runs.
    """
    run_dirs = find_run_dirs(results_dir)
    if not run_dirs:
        print("No run directories containing metrics.json were found.", file=sys.stderr)
        return pd.DataFrame(), 0

    records = []
    n_incomplete = 0
    for run_dir in run_dirs:
        try:
            record, incomplete = build_run_record(run_dir)
            records.append(record)
            if incomplete:
                n_incomplete += 1
        except Exception as exc:
            warnings.warn(f"[ERROR] Skipping {run_dir}: {exc}")
            n_incomplete += 1

    df = pd.DataFrame(records)
    # Ensure all expected columns are present (fill missing with NaN)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df, n_incomplete


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _boxplot_grouped(df, metric_col, group_col, title, ylabel, save_path):
    """Create a box-plot of *metric_col* grouped by *group_col* and save it.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str
    group_col : str
    title : str
    ylabel : str
    save_path : Path
    """
    subset = df[[group_col, metric_col]].dropna()
    if subset.empty:
        warnings.warn(f"No data for plot '{title}'; skipping.")
        return

    groups = subset.groupby(group_col)[metric_col].apply(list)
    labels = list(groups.index)
    data = [groups[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(group_col)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


def generate_plots(df, plots_dir):
    """Generate all comparison plots that have the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Summary table.
    plots_dir : Path
        Directory where plots will be saved.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    group_col = "loss_type"
    if group_col not in df.columns or df[group_col].isna().all():
        warnings.warn(
            "Column 'loss_type' is absent or all-NaN; no grouped plots will be generated."
        )
        return

    # Helper: check whether a metric column has any non-NaN data
    def _has_data(col):
        return col in df.columns and not df[col].isna().all()

    # ---- Chirp experiments -------------------------------------------------
    chirp_mask = (
        df["signal_type"].str.lower().str.contains("chirp", na=False)
        if "signal_type" in df.columns
        else pd.Series(True, index=df.index)
    )
    chirp_df = df[chirp_mask]

    if not chirp_df.empty:
        for metric, label in [
            ("ridge_error", "Ridge Error"),
            ("spread", "Spread"),
        ]:
            if _has_data(metric):
                _boxplot_grouped(
                    chirp_df, metric, group_col,
                    title=f"Chirp — {label} by {group_col}",
                    ylabel=label,
                    save_path=plots_dir / f"chirp_{metric}_by_{group_col}.png",
                )

    # ---- Gabor / multi-component experiments --------------------------------
    gabor_mask = (
        df["signal_type"].str.lower()
        .str.contains("gabor|multi", na=False, regex=True)
        if "signal_type" in df.columns
        else pd.Series(False, index=df.index)
    )
    gabor_df = df[gabor_mask]

    if not gabor_df.empty:
        for metric, label in [
            ("ambiguity_score", "Ambiguity Score"),
            ("peak_localization_error", "Peak Localization Error"),
        ]:
            if _has_data(metric):
                _boxplot_grouped(
                    gabor_df, metric, group_col,
                    title=f"Gabor/Multi \u2014 {label} by {group_col}",
                    ylabel=label,
                    save_path=plots_dir / f"gabor_{metric}_by_{group_col}.png",
                )

    # ---- All experiments ----------------------------------------------------
    for metric, label in [
        ("time_rel_error", "Time Relative Error"),
        ("freq_rel_error", "Frequency Relative Error"),
    ]:
        if _has_data(metric):
            _boxplot_grouped(
                df, metric, group_col,
                title=f"All runs — {label} by {group_col}",
                ylabel=label,
                save_path=plots_dir / f"all_{metric}_by_{group_col}.png",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate INR-TFR experiment results into a CSV summary and plots."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Root directory containing run sub-folders (default: results).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: results_dir does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning results directory: {results_dir}")

    # --- Collect runs -------------------------------------------------------
    df, n_incomplete = collect_runs(results_dir)
    n_runs = len(df)

    if n_runs == 0:
        print("No runs found.  Exiting.")
        sys.exit(0)

    # --- Save summary CSV ---------------------------------------------------
    summary_path = results_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved: {summary_path}")

    # --- Generate plots -----------------------------------------------------
    plots_dir = results_dir / "analysis_plots"
    print("Generating comparison plots…")
    generate_plots(df, plots_dir)

    # --- Final report -------------------------------------------------------
    print(
        f"\n{'=' * 52}\n"
        f"  ANALYSIS COMPLETE\n"
        f"{'=' * 52}\n"
        f"  Runs found          : {n_runs}\n"
        f"  Incomplete runs     : {n_incomplete}\n"
        f"  Summary CSV         : {summary_path}\n"
        f"  Plots folder        : {plots_dir}\n"
        f"{'=' * 52}"
    )


if __name__ == "__main__":
    main()
