#!/usr/bin/env python3
"""
dashboard.py
============
Streamlit research dashboard for INR-TFR experiments.

Launch
------
    streamlit run dashboard.py

Install dependencies (once)
---------------------------
    pip install streamlit>=1.28 pandas matplotlib numpy pyyaml torch

What the dashboard shows
------------------------
  📋 Overview       – table of all runs with key metrics and filters
  🔬 Run Detail     – config, metrics, figure, loss curve and log for one run
  📊 Compare        – side-by-side metrics and loss-curve overlays
  🚀 Launch         – configure and start a new experiment from the GUI
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="INR-TFR Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data-loading helpers (cached so repeated renders are fast)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def load_summary(results_dir: str) -> pd.DataFrame:
    """Scan *results_dir* and return one row per completed run."""
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        return pd.DataFrame()

    records = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        record: dict = {
            "run_id": run_dir.name,
            "run_path": str(run_dir),
        }

        # Config
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            try:
                with open(cfg_path) as fh:
                    cfg = yaml.safe_load(fh) or {}
                for k, v in cfg.items():
                    record[f"cfg.{k}"] = v
            except Exception:
                pass

        # Metrics
        try:
            with open(metrics_path) as fh:
                metrics = json.load(fh)
            record.update(metrics)
        except Exception:
            pass

        # File presence flags
        record["has_figure"] = (run_dir / "final_figure.png").exists()
        record["has_loss_curve"] = (run_dir / "loss_curve.csv").exists()
        record["has_log"] = (run_dir / "log.txt").exists()

        records.append(record)

    return pd.DataFrame(records) if records else pd.DataFrame()


@st.cache_data(ttl=10)
def load_loss_curve(run_path: str) -> pd.DataFrame:
    """Load loss_curve.csv for one run; return empty DataFrame on failure."""
    path = Path(run_path) / "loss_curve.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=10)
def load_log(run_path: str) -> str:
    """Return the content of log.txt for one run."""
    path = Path(run_path) / "log.txt"
    if path.exists():
        try:
            return path.read_text()
        except Exception:
            pass
    return ""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔬 INR-TFR")
    st.caption("Research Dashboard")
    st.divider()

    results_dir = st.text_input(
        "Results directory",
        value="results",
        help="Path to the folder that contains your experiment run sub-folders.",
    )

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    df_all = load_summary(results_dir)
    n_runs = len(df_all)
    st.metric("Runs found", n_runs)

    if n_runs > 0 and "loss_type" in df_all.columns:
        st.write("**By loss type**")
        for lt, cnt in df_all["loss_type"].value_counts().items():
            st.write(f"• {lt}: **{cnt}**")


# ---------------------------------------------------------------------------
# Main tab bar
# ---------------------------------------------------------------------------

tab_overview, tab_detail, tab_compare, tab_launch = st.tabs([
    "📋 Overview",
    "🔬 Run Detail",
    "📊 Compare",
    "🚀 Launch Experiment",
])


# ===========================================================================
# Tab 1 — Overview
# ===========================================================================

with tab_overview:
    st.header("All Experiment Runs")

    if df_all.empty:
        st.info(
            f"No runs found in **{results_dir}/**.\n\n"
            "Go to the **🚀 Launch Experiment** tab to run your first "
            "experiment, or check that your results directory contains "
            "sub-folders with a `metrics.json` file."
        )
    else:
        # ----- Filters ------------------------------------------------------
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            loss_filter: list = []
            if "loss_type" in df_all.columns:
                loss_filter = st.multiselect(
                    "Filter by loss type",
                    options=df_all["loss_type"].dropna().unique().tolist(),
                    default=[],
                )
        with col_f2:
            sig_filter: list = []
            if "signal_type" in df_all.columns:
                sig_filter = st.multiselect(
                    "Filter by signal type",
                    options=df_all["signal_type"].dropna().unique().tolist(),
                    default=[],
                )

        df_show = df_all.copy()
        if loss_filter and "loss_type" in df_show.columns:
            df_show = df_show[df_show["loss_type"].isin(loss_filter)]
        if sig_filter and "signal_type" in df_show.columns:
            df_show = df_show[df_show["signal_type"].isin(sig_filter)]

        # ----- Display table ------------------------------------------------
        display_cols = ["run_id"]
        for col in [
            "loss_type", "signal_type",
            "cfg.N", "cfg.k", "cfg.sigma", "cfg.lr",
            "cfg.num_epochs", "cfg.lambda_moment", "cfg.omega0", "cfg.seed",
            "time_rel_error", "freq_rel_error", "final_loss",
            "runtime_seconds",
        ]:
            if col in df_show.columns:
                display_cols.append(col)

        st.dataframe(
            df_show[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

        # ----- Best-run highlights ------------------------------------------
        st.subheader("🏆 Best Runs")
        col1, col2, col3 = st.columns(3)

        def _best_metric(col_name, label, col_widget):
            if col_name in df_show.columns and \
                    not df_show[col_name].isna().all():
                best = df_show.loc[df_show[col_name].idxmin()]
                col_widget.metric(
                    label,
                    f"{best[col_name]:.4f}",
                    best["run_id"],
                )

        _best_metric("time_rel_error", "↓ Lowest time rel error", col1)
        _best_metric("freq_rel_error", "↓ Lowest freq rel error", col2)

        if "final_loss" in df_show.columns and \
                not df_show["final_loss"].isna().all():
            best_l = df_show.loc[df_show["final_loss"].idxmin()]
            col3.metric(
                "↓ Lowest final loss",
                f"{best_l['final_loss']:.4e}",
                best_l["run_id"],
            )


# ===========================================================================
# Tab 2 — Run Detail
# ===========================================================================

with tab_detail:
    st.header("Run Detail")

    if df_all.empty:
        st.info("No runs found yet.")
    else:
        selected_run = st.selectbox(
            "Select a run",
            options=df_all["run_id"].tolist(),
        )

        if selected_run:
            row = df_all[df_all["run_id"] == selected_run].iloc[0]
            run_path = row["run_path"]

            # ---- Key metrics row -------------------------------------------
            col1, col2, col3, col4 = st.columns(4)
            def _show_metric(widget, key, label, fmt):
                val = row.get(key)
                if val is not None and not (
                        isinstance(val, float) and np.isnan(val)):
                    widget.metric(label, fmt.format(val))

            _show_metric(col1, "time_rel_error", "Time rel error", "{:.6f}")
            _show_metric(col2, "freq_rel_error", "Freq rel error", "{:.6f}")
            _show_metric(col3, "final_loss",     "Final loss",     "{:.4e}")
            _show_metric(col4, "runtime_seconds","Runtime",        "{:.1f}s")

            st.divider()

            # ---- Config + metrics side by side -----------------------------
            col_cfg, col_met = st.columns(2)

            with col_cfg:
                st.subheader("⚙️ Configuration")
                cfg_path = Path(run_path) / "config.yaml"
                if cfg_path.exists():
                    try:
                        with open(cfg_path) as fh:
                            cfg_data = yaml.safe_load(fh) or {}
                        st.json(cfg_data)
                    except Exception as exc:
                        st.warning(f"Could not load config.yaml: {exc}")
                else:
                    cfg_cols = {
                        k.replace("cfg.", ""): v
                        for k, v in row.items()
                        if k.startswith("cfg.") and not (
                            isinstance(v, float) and np.isnan(v))
                    }
                    if cfg_cols:
                        st.json(cfg_cols)
                    else:
                        st.warning("config.yaml not found for this run.")

            with col_met:
                st.subheader("📈 Final Metrics")
                metrics_path = Path(run_path) / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as fh:
                            st.json(json.load(fh))
                    except Exception as exc:
                        st.warning(f"Could not load metrics.json: {exc}")
                else:
                    st.warning("metrics.json not found for this run.")

            st.divider()

            # ---- Output figure ---------------------------------------------
            figure_path = Path(run_path) / "final_figure.png"
            if figure_path.exists():
                st.subheader("🖼️ Output Figure")
                st.image(str(figure_path), use_container_width=True)
            else:
                st.info("No figure saved for this run.")

            # ---- Loss curve ------------------------------------------------
            st.subheader("📉 Loss Curve")
            loss_df = load_loss_curve(run_path)
            if not loss_df.empty and "epoch" in loss_df.columns:
                available = [
                    c for c in [
                        "total_loss", "marginal_loss", "moment_loss",
                        "time_rel_error", "freq_rel_error",
                    ]
                    if c in loss_df.columns
                ]
                if available:
                    selected_metrics = st.multiselect(
                        "Metrics to show",
                        options=available,
                        default=available[:3],
                        key="detail_metrics",
                    )
                    if selected_metrics:
                        st.line_chart(
                            loss_df.set_index("epoch")[selected_metrics],
                            use_container_width=True,
                        )
            else:
                st.info(
                    "No loss curve data (loss_curve.csv not found). "
                    "Only runs launched via run_experiment.py save this file."
                )

            # ---- Training log ----------------------------------------------
            st.subheader("📝 Training Log")
            log_content = load_log(run_path)
            if log_content:
                with st.expander("Show full log", expanded=False):
                    st.code(log_content, language=None)
            else:
                st.info("No log.txt found for this run.")


# ===========================================================================
# Tab 3 — Compare
# ===========================================================================

with tab_compare:
    st.header("Compare Runs")

    if df_all.empty:
        st.info("No runs found yet.")
    elif len(df_all) < 2:
        st.info(
            "You need at least 2 completed runs to compare. "
            "Launch more experiments in the **🚀 Launch Experiment** tab."
        )
    else:
        run_ids = df_all["run_id"].tolist()
        selected_runs = st.multiselect(
            "Select runs to compare",
            options=run_ids,
            default=run_ids[:min(4, len(run_ids))],
        )

        if len(selected_runs) < 2:
            st.info("Select at least 2 runs.")
        else:
            compare_df = df_all[df_all["run_id"].isin(selected_runs)].copy()

            # ---- Metric bar charts -----------------------------------------
            metric_options = [
                c for c in [
                    "time_rel_error", "freq_rel_error",
                    "final_loss", "final_marginal_loss",
                    "final_moment_loss", "runtime_seconds",
                ]
                if c in compare_df.columns
                and not compare_df[c].isna().all()
            ]

            if metric_options:
                st.subheader("📊 Metric Comparison")
                n_cols = min(3, len(metric_options))
                cols = st.columns(n_cols)
                for i, metric in enumerate(metric_options[:6]):
                    with cols[i % n_cols]:
                        chart_data = (
                            compare_df[["run_id", metric]]
                            .dropna()
                            .set_index("run_id")
                        )
                        st.write(f"**{metric}**")
                        st.bar_chart(chart_data, use_container_width=True)

            # ---- Loss-curve overlay ----------------------------------------
            st.subheader("📉 Loss Curve Overlay")
            overlay_metric = st.selectbox(
                "Metric to overlay",
                options=[
                    "total_loss", "marginal_loss", "moment_loss",
                    "time_rel_error", "freq_rel_error",
                ],
                index=0,
            )
            overlay_data: dict = {}
            for rid in selected_runs:
                rpath = df_all[df_all["run_id"] == rid]["run_path"].values[0]
                ldf = load_loss_curve(rpath)
                if not ldf.empty and overlay_metric in ldf.columns \
                        and "epoch" in ldf.columns:
                    overlay_data[rid] = ldf.set_index("epoch")[overlay_metric]

            if overlay_data:
                overlay_df = pd.DataFrame(overlay_data)
                st.line_chart(overlay_df, use_container_width=True)
            else:
                st.info(
                    "No loss curve data for the selected runs. "
                    "loss_curve.csv is only saved by run_experiment.py."
                )

            # ---- Side-by-side figures --------------------------------------
            st.subheader("🖼️ Output Figures")
            fig_cols = st.columns(len(selected_runs))
            for i, rid in enumerate(selected_runs):
                rpath = df_all[df_all["run_id"] == rid]["run_path"].values[0]
                fig_path = Path(rpath) / "final_figure.png"
                with fig_cols[i]:
                    st.write(f"**{rid}**")
                    if fig_path.exists():
                        st.image(str(fig_path), use_container_width=True)
                    else:
                        st.write("_(no figure)_")


# ===========================================================================
# Tab 4 — Launch Experiment
# ===========================================================================

with tab_launch:
    st.header("Launch New Experiment")
    st.write(
        "Configure your experiment below and click **Launch**. "
        "Results are saved automatically and will appear in the other "
        "tabs after you press **🔄 Refresh** in the sidebar."
    )

    # Locate run_experiment.py
    this_dir = Path(__file__).parent
    runner_path = this_dir / "run_experiment.py"

    if not runner_path.exists():
        st.error(
            f"`run_experiment.py` not found at `{runner_path}`. "
            "Make sure both scripts are in the same directory."
        )
    else:
        with st.form("launch_form"):

            # ---- Signal ------------------------------------------------
            st.subheader("📡 Signal")
            col1, col2, col3 = st.columns(3)

            with col1:
                N = st.select_slider(
                    "Resolution (N×N)",
                    options=[64, 128, 256],
                    value=128,
                    help="Grid resolution. Higher = more detail but slower "
                         "to train.",
                )
                k = st.slider(
                    "Chirp rate (k)",
                    min_value=10.0, max_value=80.0,
                    value=40.0, step=1.0,
                    help="How fast the frequency sweeps over time. "
                         "Higher = faster sweep.",
                )

            with col2:
                sigma = st.slider(
                    "Window width (σ)",
                    min_value=0.05, max_value=0.35,
                    value=0.12, step=0.01,
                    help="Width of the Gaussian envelope around the chirp. "
                         "Smaller = shorter burst.",
                )
                t0 = st.slider(
                    "Envelope centre (t₀)",
                    min_value=-0.4, max_value=0.4,
                    value=0.0, step=0.05,
                    help="Time position of the chirp peak. "
                         "0 = centred in the window.",
                )

            with col3:
                omega0 = st.select_slider(
                    "SIREN ω₀",
                    options=[10.0, 20.0, 30.0, 40.0, 50.0],
                    value=30.0,
                    help="Frequency multiplier for the SIREN network. "
                         "30 is the recommended default.",
                )

            # ---- Training ----------------------------------------------
            st.subheader("🏋️ Training")
            col4, col5, col6 = st.columns(3)

            with col4:
                loss_type = st.radio(
                    "Loss type",
                    options=["marginal_only", "marginals_plus_moments"],
                    format_func=lambda x: (
                        "Marginals only"
                        if x == "marginal_only"
                        else "Marginals + Moments"
                    ),
                    help=(
                        "**Marginals only** is the baseline — the network "
                        "matches 1D time and frequency distributions.\n\n"
                        "**Marginals + Moments** also enforces time-frequency "
                        "correlation constraints, which helps recover the "
                        "chirp's diagonal ridge structure."
                    ),
                )
                lr = st.select_slider(
                    "Learning rate",
                    options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    value=1e-4,
                    format_func=lambda x: f"{x:.0e}",
                    help="Step size for the Adam optimiser. 1e-4 is a safe "
                         "default.",
                )

            with col5:
                num_epochs = st.slider(
                    "Training epochs",
                    min_value=500, max_value=5000,
                    value=2000, step=100,
                    help="More epochs = better convergence, but takes longer.",
                )
                seed = st.number_input(
                    "Random seed",
                    min_value=0, max_value=99999,
                    value=42,
                    help="Use the same seed to reproduce a run exactly.",
                )

            with col6:
                lambda_moment = st.slider(
                    "Moment weight (λ)",
                    min_value=0.1, max_value=10.0,
                    value=1.0, step=0.1,
                    disabled=(loss_type == "marginal_only"),
                    help="How strongly moment constraints are enforced "
                         "relative to marginal constraints. "
                         "Only active with Marginals + Moments.",
                )

            # ---- Output ------------------------------------------------
            st.subheader("💾 Output")
            col7, col8 = st.columns(2)
            with col7:
                out_results_dir = st.text_input(
                    "Results directory",
                    value=results_dir,
                    help="Where to save this run.",
                )
            with col8:
                custom_run_id = st.text_input(
                    "Run name (optional)",
                    value="",
                    placeholder="auto-generated from timestamp if empty",
                    help="Give this run a memorable name such as "
                         "'chirp_fast_k60' so you can find it easily later.",
                )

            submitted = st.form_submit_button(
                "🚀 Launch Experiment",
                use_container_width=True,
                type="primary",
            )

        # ---- Handle form submission ------------------------------------
        if submitted:
            cmd = [
                sys.executable, str(runner_path),
                "--signal_type", "chirp",
                "--N", str(N),
                "--k", str(k),
                "--sigma", str(sigma),
                "--t0", str(t0),
                "--omega0", str(omega0),
                "--loss_type", loss_type,
                "--lr", str(lr),
                "--num_epochs", str(num_epochs),
                "--lambda_moment", str(lambda_moment),
                "--seed", str(int(seed)),
                "--results_dir", out_results_dir,
            ]
            if custom_run_id.strip():
                cmd += ["--run_id", custom_run_id.strip()]

            with st.expander("Command being run", expanded=False):
                st.code(" ".join(cmd), language="bash")

            with st.status("⚙️ Training in progress…", expanded=True) as status:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=str(this_dir),
                    )
                    if result.returncode == 0:
                        status.update(
                            label="✅ Experiment complete!",
                            state="complete",
                            expanded=False,
                        )
                        st.success("Training finished successfully!")
                        with st.expander("📜 Training output", expanded=False):
                            st.code(result.stdout, language=None)
                        st.cache_data.clear()
                        st.info(
                            "Press **🔄 Refresh** in the sidebar to see "
                            "this run in the Overview, Detail and Compare tabs."
                        )
                    else:
                        status.update(
                            label="❌ Experiment failed",
                            state="error",
                            expanded=True,
                        )
                        st.error(
                            "Training failed — see the error output below."
                        )
                        st.code(result.stderr or result.stdout, language=None)

                except FileNotFoundError as exc:
                    status.update(
                        label="❌ Script not found",
                        state="error",
                    )
                    st.error(f"Could not start run_experiment.py: {exc}")

                except Exception as exc:
                    status.update(
                        label="❌ Unexpected error",
                        state="error",
                    )
                    st.error(f"Unexpected error: {exc}")
