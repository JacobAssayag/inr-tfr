#!/usr/bin/env python3
"""
run_experiment.py
=================
Unified, parameterised INR-TFR experiment runner.

Combines the logic of marginal_experiment.py and moment_experiment.py into a
single script with fully configurable hyperparameters.  Every run is saved to
its own timestamped sub-directory so results are always reproducible and
comparable side-by-side.

Usage
-----
    python run_experiment.py
    python run_experiment.py --loss_type marginals_plus_moments --lambda_moment 5.0
    python run_experiment.py --k 60 --sigma 0.08 --num_epochs 3000 --seed 7
    python run_experiment.py --help

Saved artefacts (per run)
--------------------------
    results/<run_id>/
        config.yaml       – all hyperparameters used
        metrics.json      – final evaluation metrics
        loss_curve.csv    – per-epoch training history
        final_figure.png  – visualisation panels
        log.txt           – full training log
"""

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_windowed_chirp(N, k, sigma, t0, device="cpu"):
    """Generate a Gaussian-windowed linear chirp and normalised marginals.

    Parameters
    ----------
    N      : int   – number of samples
    k      : float – chirp rate
    sigma  : float – Gaussian window width
    t0     : float – envelope centre
    device : str

    Returns
    -------
    t_grid      : Tensor (N,) – time axis in [-0.5, 0.5]
    x           : Tensor (N,) complex – analytic signal
    time_energy : Tensor (N,) – normalised |x(t)|², sums to 1
    freq_energy : Tensor (N,) – normalised |X(f)|², DC-centred, sums to 1
    """
    t_grid = torch.linspace(-0.5, 0.5, N, device=device)
    x = (
        torch.exp(-(t_grid - t0) ** 2 / (2 * sigma ** 2))
        * torch.exp(1j * torch.pi * k * t_grid ** 2)
    )

    raw_time = x.abs().pow(2)
    X = torch.fft.fft(x, norm="ortho")
    raw_freq = X.abs().pow(2)

    parseval_err = (raw_time.sum() - raw_freq.sum()).abs() / raw_time.sum()
    assert parseval_err < 1e-3, f"Parseval violation: {parseval_err:.2e}"
    assert raw_time.min() >= 0
    assert raw_freq.min() >= 0
    assert torch.isfinite(raw_time).all()
    assert torch.isfinite(raw_freq).all()

    time_energy = raw_time / raw_time.sum()
    freq_energy = torch.fft.fftshift(raw_freq) / raw_freq.sum()

    assert (time_energy.sum() - 1.0).abs() < 1e-6
    assert (freq_energy.sum() - 1.0).abs() < 1e-6

    return t_grid, x, time_energy, freq_energy


# ---------------------------------------------------------------------------
# SIREN model
# ---------------------------------------------------------------------------

def _init_siren(linear, is_first, omega0):
    fan_in = linear.weight.size(1)
    bound = 1.0 / fan_in if is_first else math.sqrt(6.0 / fan_in) / omega0
    nn.init.uniform_(linear.weight, -bound, bound)
    nn.init.uniform_(linear.bias, -1.0 / math.sqrt(fan_in),
                     1.0 / math.sqrt(fan_in))


class TFD_Network(nn.Module):
    """3-hidden-layer SIREN that maps (t, f) coordinates to energy density."""

    def __init__(self, omega0=30.0):
        super().__init__()
        self.omega0 = omega0
        self.hidden1 = nn.Linear(2, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        _init_siren(self.hidden1, True, omega0)
        _init_siren(self.hidden2, False, omega0)
        _init_siren(self.hidden3, False, omega0)

    def forward(self, coords):
        out = torch.sin(self.omega0 * self.hidden1(coords))
        out = torch.sin(self.omega0 * self.hidden2(out))
        out = torch.sin(self.omega0 * self.hidden3(out))
        return F.softplus(self.output(out)).squeeze(-1)


# ---------------------------------------------------------------------------
# Moment helpers
# ---------------------------------------------------------------------------

def compute_moments(P, TT, FF):
    """Return (mean_t, mean_f, var_t, var_f, cov_tf) for distribution P."""
    mean_t = (TT * P).sum()
    mean_f = (FF * P).sum()
    var_t = ((TT - mean_t) ** 2 * P).sum()
    var_f = ((FF - mean_f) ** 2 * P).sum()
    cov_tf = ((TT - mean_t) * (FF - mean_f) * P).sum()
    return mean_t, mean_f, var_t, var_f, cov_tf


# ---------------------------------------------------------------------------
# Cohen-class reference (smoothed pseudo-WVD)
# ---------------------------------------------------------------------------

def compute_pWVD(x, N, device):
    """Compute frequency-centred smoothed pseudo-Wigner-Ville Distribution."""
    x_padded = torch.zeros(2 * N, dtype=torch.complex64, device=device)
    x_padded[:N] = x
    half = N // 2
    tau_range = torch.arange(-half, half, device=device)

    WVD = torch.zeros(N, N, device=device)
    for t in range(N):
        t_plus = (t + tau_range) % (2 * N)
        t_minus = (t - tau_range) % (2 * N)
        row = torch.fft.fft(
            x_padded[t_plus] * x_padded[t_minus].conj(), norm="ortho"
        )
        WVD[t, :] = row.real

    WVD = torch.fft.fftshift(WVD, dim=1)

    ks, sig = 15, 2.0
    idx = torch.arange(ks, dtype=torch.float32, device=device)
    g1d = torch.exp(-0.5 * ((idx - ks // 2) / sig) ** 2)
    g1d /= g1d.sum()
    kernel = torch.outer(g1d, g1d)
    kernel /= kernel.sum()

    pWVD = F.conv2d(
        WVD.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=ks // 2,
    ).squeeze()
    pWVD = torch.clamp(pWVD, min=0.0)
    pWVD /= pWVD.sum()

    assert pWVD.min() >= 0.0
    assert abs(pWVD.sum().item() - 1.0) < 1e-5

    return pWVD


# ---------------------------------------------------------------------------
# Results figure
# ---------------------------------------------------------------------------

def _save_figure(run_dir, cfg, t_grid, f_norm, time_energy, freq_energy,
                 pred_time, pred_freq, P_final, pWVD, history, loss_type,
                 rel_err_t, rel_err_f, tgt_moments=None):
    """Save a 2×3 results figure to run_dir/final_figure.png."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    title = (
        f"INR-TFR  |  {cfg['loss_type']}  |  "
        f"k={cfg['k']}  σ={cfg['sigma']}  "
        f"lr={cfg['lr']}  λ_mom={cfg['lambda_moment']}  "
        f"epochs={cfg['num_epochs']}"
    )
    fig.suptitle(title, fontsize=10)

    t_cpu = t_grid.cpu()
    f_cpu = f_norm.cpu()

    # (0,0) — time marginal
    ax = axes[0, 0]
    ax.plot(t_cpu, time_energy.cpu(), color="steelblue", lw=1.5, label="True")
    ax.plot(t_cpu, pred_time.detach().cpu(), "--",
            color="firebrick", lw=1.5, label="Predicted")
    ax.set_title(f"Time marginal  (rel err = {rel_err_t:.4f})", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("t")

    # (1,0) — freq marginal
    ax = axes[1, 0]
    ax.plot(f_cpu, freq_energy.cpu(), color="steelblue", lw=1.5, label="True")
    ax.plot(f_cpu, pred_freq.detach().cpu(), "--",
            color="firebrick", lw=1.5, label="Predicted")
    ax.set_title(f"Freq marginal  (rel err = {rel_err_f:.4f})", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("f (normalised)")

    # (0,1) — learned P(t,f)
    P_np = P_final.detach().cpu().numpy()
    ax = axes[0, 1]
    label = ("Learned P(t,f) — Marginals + Moments"
             if loss_type == "marginals_plus_moments"
             else "Learned P(t,f) — Marginals Only")
    im = ax.imshow(P_np.T, origin="lower", aspect="auto", cmap="viridis",
                   vmin=np.percentile(P_np, 1), vmax=np.percentile(P_np, 99))
    ax.set_xlabel("Time index")
    ax.set_ylabel("Frequency index")
    ax.set_title(label, fontsize=9)
    plt.colorbar(im, ax=ax, label="Energy density")

    # (1,1) — Cohen-class reference
    pWVD_np = pWVD.cpu().numpy()
    ax = axes[1, 1]
    im2 = ax.imshow(pWVD_np.T, origin="lower", aspect="auto", cmap="viridis",
                    vmin=np.percentile(pWVD_np, 1),
                    vmax=np.percentile(pWVD_np, 99))
    ax.set_xlabel("Time index")
    ax.set_ylabel("Frequency index")
    ax.set_title("Cohen-Class Reference (smoothed pWVD)", fontsize=9)
    plt.colorbar(im2, ax=ax, label="Energy density")

    if history:
        epochs_ax = [h["epoch"] for h in history]

        # (0,2) — loss convergence
        ax = axes[0, 2]
        ax.semilogy(epochs_ax, [h["marginal_loss"] for h in history],
                    label="Marginal", color="steelblue", lw=1.2)
        if loss_type == "marginals_plus_moments":
            ax.semilogy(epochs_ax, [h["moment_loss"] for h in history],
                        label="Moment", color="firebrick", lw=1.2)
            ax.semilogy(epochs_ax, [h["total_loss"] for h in history],
                        label="Total", color="black", lw=1.2, ls="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Loss convergence", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (1,2) — moment tracking or relative error curves
        ax = axes[1, 2]
        if loss_type == "marginals_plus_moments" and tgt_moments is not None:
            tgt_mean_t, tgt_mean_f, _, _, tgt_cov_tf = tgt_moments
            ax.plot(epochs_ax, [h["cov_tf"] for h in history],
                    label=f"cov_tf  (tgt={tgt_cov_tf.item():.4f})",
                    color="darkorange", lw=1.5)
            ax.axhline(tgt_cov_tf.item(), color="darkorange", ls=":", alpha=0.7)
            ax.plot(epochs_ax, [h["mean_t"] for h in history],
                    label=f"mean_t  (tgt={tgt_mean_t.item():.4f})",
                    color="steelblue", lw=1.2)
            ax.axhline(tgt_mean_t.item(), color="steelblue", ls=":", alpha=0.7)
            ax.plot(epochs_ax, [h["mean_f"] for h in history],
                    label=f"mean_f  (tgt={tgt_mean_f.item():.4f})",
                    color="seagreen", lw=1.2)
            ax.axhline(tgt_mean_f.item(), color="seagreen", ls=":", alpha=0.7)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title("Moment tracking over training", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        else:
            ax.plot(epochs_ax, [h["time_rel_error"] for h in history],
                    label="rel_err_t", color="steelblue", lw=1.5)
            ax.plot(epochs_ax, [h["freq_rel_error"] for h in history],
                    label="rel_err_f", color="firebrick", lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Relative error")
            ax.set_title("Marginal relative errors", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    else:
        axes[0, 2].axis("off")
        axes[1, 2].axis("off")

    plt.tight_layout()
    fig_path = run_dir / "final_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run(cfg, run_dir):
    """Execute one experiment defined by cfg; save all artefacts to run_dir.

    Parameters
    ----------
    cfg     : dict – all hyperparameters (see build_parser for keys)
    run_dir : Path – destination directory (created if absent)

    Returns
    -------
    dict – final metrics
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.txt"
    log_fh = open(log_path, "w")

    def log(msg=""):
        print(msg, flush=True)
        print(msg, file=log_fh, flush=True)

    try:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        log(f"[run_experiment] run_id  : {run_dir.name}")
        log(f"[run_experiment] device  : {device}")
        log(f"[run_experiment] loss    : {cfg['loss_type']}")
        log()

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        if device == "cuda":
            torch.backends.cudnn.deterministic = True

        N = cfg["N"]
        loss_type = cfg["loss_type"]
        lambda_moment = cfg["lambda_moment"]

        # ---- Signal --------------------------------------------------------
        t_grid, x, time_energy, freq_energy = generate_windowed_chirp(
            N, cfg["k"], cfg["sigma"], cfg["t0"], device
        )

        # ---- Coordinate grid -----------------------------------------------
        t_norm = torch.linspace(-1.0, 1.0, N, device=device)
        f_norm = torch.linspace(-1.0, 1.0, N, device=device)
        TT, FF = torch.meshgrid(t_norm, f_norm, indexing="ij")
        coords = torch.stack([TT.reshape(-1), FF.reshape(-1)], dim=-1)

        # ---- Cohen-class reference -----------------------------------------
        log("Computing Cohen-class reference (pWVD)…")
        pWVD = compute_pWVD(x, N, device)
        tgt_mean_t, tgt_mean_f, tgt_var_t, tgt_var_f, tgt_cov_tf = \
            compute_moments(pWVD, TT, FF)

        if loss_type == "marginals_plus_moments":
            log("Target moments (from pWVD):")
            log(f"  mean_t={tgt_mean_t.item():.6f}  "
                f"mean_f={tgt_mean_f.item():.6f}")
            log(f"  var_t={tgt_var_t.item():.6f}   "
                f"var_f={tgt_var_f.item():.6f}")
            log(f"  cov_tf={tgt_cov_tf.item():.6f}")
            log()

        # ---- Model ---------------------------------------------------------
        net = TFD_Network(omega0=cfg["omega0"]).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

        num_epochs = cfg["num_epochs"]
        log_every = cfg["log_every"]

        history = []
        t_start = time.time()

        log(f"Training for {num_epochs} epochs…")

        # ---- Training loop -------------------------------------------------
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            pred_flat = net(coords)
            P = pred_flat.view(N, N)
            P = P / (P.sum() + 1e-8)

            pred_time = P.sum(dim=1)
            pred_freq = P.sum(dim=0)

            marginal_loss = (
                F.mse_loss(pred_time, time_energy)
                + F.mse_loss(pred_freq, freq_energy)
            )

            if loss_type == "marginals_plus_moments":
                mean_t, mean_f, var_t, var_f, cov_tf = \
                    compute_moments(P, TT, FF)
                moment_loss = (
                    (mean_t - tgt_mean_t) ** 2
                    + (mean_f - tgt_mean_f) ** 2
                    + (var_t - tgt_var_t) ** 2
                    + (var_f - tgt_var_f) ** 2
                    + (cov_tf - tgt_cov_tf) ** 2
                )
                total_loss = marginal_loss + lambda_moment * moment_loss
            else:
                moment_loss = torch.tensor(0.0, device=device)
                cov_tf = torch.tensor(0.0, device=device)
                mean_t = torch.tensor(0.0, device=device)
                mean_f = torch.tensor(0.0, device=device)
                total_loss = marginal_loss

            total_loss.backward()
            optimizer.step()

            assert torch.isfinite(P).all(), "NaN/Inf in P"
            assert torch.isfinite(total_loss), "NaN/Inf in loss"

            # Record history every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    rel_t = (pred_time - time_energy).norm() / \
                        time_energy.norm()
                    rel_f = (pred_freq - freq_energy).norm() / \
                        freq_energy.norm()
                history.append({
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "marginal_loss": marginal_loss.item(),
                    "moment_loss": moment_loss.item(),
                    "time_rel_error": rel_t.item(),
                    "freq_rel_error": rel_f.item(),
                    "cov_tf": cov_tf.item(),
                    "mean_t": mean_t.item(),
                    "mean_f": mean_f.item(),
                })

            # Console / log output
            if epoch % log_every == 0:
                with torch.no_grad():
                    rel_t_log = (pred_time - time_energy).norm() / \
                        time_energy.norm()
                    rel_f_log = (pred_freq - freq_energy).norm() / \
                        freq_energy.norm()
                if loss_type == "marginals_plus_moments":
                    log(
                        f"Epoch {epoch:5d}/{num_epochs} | "
                        f"Total={total_loss.item():.3e} | "
                        f"Marginal={marginal_loss.item():.3e} | "
                        f"Moment={moment_loss.item():.3e} | "
                        f"cov_tf={cov_tf.item():.4f} | "
                        f"rel_t={rel_t_log.item():.4f} | "
                        f"rel_f={rel_f_log.item():.4f}"
                    )
                else:
                    log(
                        f"Epoch {epoch:5d}/{num_epochs} | "
                        f"Loss={total_loss.item():.3e} | "
                        f"rel_t={rel_t_log.item():.4f} | "
                        f"rel_f={rel_f_log.item():.4f}"
                    )

        runtime = time.time() - t_start
        log(f"\nTraining complete in {runtime:.1f}s")

        # ---- Final evaluation ----------------------------------------------
        with torch.no_grad():
            pred_flat = net(coords)
            P_final = pred_flat.view(N, N)
            P_final = P_final / (P_final.sum() + 1e-8)
            pred_time_f = P_final.sum(dim=1)
            pred_freq_f = P_final.sum(dim=0)
            rel_err_t = (pred_time_f - time_energy).norm() / time_energy.norm()
            rel_err_f = (pred_freq_f - freq_energy).norm() / freq_energy.norm()
            marginal_loss_f = (
                F.mse_loss(pred_time_f, time_energy)
                + F.mse_loss(pred_freq_f, freq_energy)
            )
            if loss_type == "marginals_plus_moments":
                mean_t_f, mean_f_f, var_t_f, var_f_f, cov_tf_f = \
                    compute_moments(P_final, TT, FF)
                moment_loss_f = (
                    (mean_t_f - tgt_mean_t) ** 2
                    + (mean_f_f - tgt_mean_f) ** 2
                    + (var_t_f - tgt_var_t) ** 2
                    + (var_f_f - tgt_var_f) ** 2
                    + (cov_tf_f - tgt_cov_tf) ** 2
                )
                total_loss_f = marginal_loss_f + lambda_moment * moment_loss_f
            else:
                total_loss_f = marginal_loss_f

        # ---- Metrics dict --------------------------------------------------
        metrics = {
            "signal_type": cfg["signal_type"],
            "loss_type": loss_type,
            "time_rel_error": round(rel_err_t.item(), 8),
            "freq_rel_error": round(rel_err_f.item(), 8),
            "final_loss": round(total_loss_f.item(), 10),
            "final_marginal_loss": round(marginal_loss_f.item(), 10),
            "mass": round(P_final.sum().item(), 8),
            "runtime_seconds": round(runtime, 2),
        }
        if loss_type == "marginals_plus_moments":
            metrics.update({
                "final_moment_loss": round(moment_loss_f.item(), 10),
                "learned_mean_t": round(mean_t_f.item(), 6),
                "learned_mean_f": round(mean_f_f.item(), 6),
                "learned_var_t": round(var_t_f.item(), 6),
                "learned_var_f": round(var_f_f.item(), 6),
                "learned_cov_tf": round(cov_tf_f.item(), 6),
                "target_mean_t": round(tgt_mean_t.item(), 6),
                "target_mean_f": round(tgt_mean_f.item(), 6),
                "target_var_t": round(tgt_var_t.item(), 6),
                "target_var_f": round(tgt_var_f.item(), 6),
                "target_cov_tf": round(tgt_cov_tf.item(), 6),
            })

        log()
        log("=" * 55)
        log("  RESULTS")
        log("=" * 55)
        log(f"  Time rel error  : {metrics['time_rel_error']:.6f}")
        log(f"  Freq rel error  : {metrics['freq_rel_error']:.6f}")
        log(f"  Final loss      : {metrics['final_loss']:.4e}")
        log(f"  Runtime         : {metrics['runtime_seconds']:.1f}s")
        log("=" * 55)

        # ---- Save artefacts ------------------------------------------------
        with open(run_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        if history:
            with open(run_dir / "loss_curve.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh,
                                        fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

        tgt_moments = (tgt_mean_t, tgt_mean_f, tgt_var_t,
                       tgt_var_f, tgt_cov_tf)
        _save_figure(
            run_dir, cfg, t_grid, f_norm, time_energy, freq_energy,
            pred_time_f, pred_freq_f, P_final, pWVD, history, loss_type,
            rel_err_t, rel_err_f,
            tgt_moments=tgt_moments,
        )

        log(f"\nAll results saved to: {run_dir}")
        return metrics

    finally:
        log_fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Run one INR-TFR experiment and save structured results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Signal
    p.add_argument("--signal_type", default="chirp", choices=["chirp"],
                   help="Signal type.")
    p.add_argument("--N", type=int, default=128,
                   help="Grid resolution (N×N).")
    p.add_argument("--k", type=float, default=40.0,
                   help="Chirp rate.")
    p.add_argument("--sigma", type=float, default=0.12,
                   help="Gaussian window width.")
    p.add_argument("--t0", type=float, default=0.0,
                   help="Chirp envelope centre.")

    # Model
    p.add_argument("--omega0", type=float, default=30.0,
                   help="SIREN frequency multiplier.")

    # Training
    p.add_argument("--loss_type", default="marginal_only",
                   choices=["marginal_only", "marginals_plus_moments"],
                   help="Loss function type.")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate.")
    p.add_argument("--num_epochs", type=int, default=2000,
                   help="Number of training epochs.")
    p.add_argument("--lambda_moment", type=float, default=1.0,
                   help="Moment loss weight "
                        "(only used with marginals_plus_moments).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    p.add_argument("--log_every", type=int, default=200,
                   help="Log progress every N epochs.")

    # Output
    p.add_argument("--results_dir", default="results",
                   help="Root directory for results.")
    p.add_argument("--run_id", default=None,
                   help="Run identifier "
                        "(auto-generated from timestamp if omitted).")

    # Config-file shortcut (used by the dashboard to avoid passing user
    # strings on the command line)
    p.add_argument("--config-file", dest="config_file", default=None,
                   help="Path to a YAML file whose keys override all other "
                        "flags.  Used internally by dashboard.py.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = vars(args)

    # If a config file was supplied, load it and let it override CLI defaults
    if cfg.get("config_file"):
        with open(cfg["config_file"]) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update({k: v for k, v in file_cfg.items() if k in cfg})
        cfg.pop("config_file", None)

    # Auto-generate run_id if not provided
    if cfg["run_id"] is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["run_id"] = f"run_{ts}_{cfg['loss_type']}"

    results_dir = Path(cfg["results_dir"]).expanduser().resolve()
    run_dir = results_dir / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config immediately so the run is identifiable even if training fails
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=True)
    print(f"Config saved: {config_path}")

    run(cfg, run_dir)
    print(f"\nRun complete: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
