"""
train.py
========
Config-driven experiment runner for INR time-frequency experiments.

Usage
-----
    from inr_tfr.train import run_experiment
    metrics = run_experiment(config)

Or via CLI:
    python run_experiment.py --config configs/marginal_only.yaml

Output layout
-------------
    results/<timestamp>_<signal>_<condition>/
        config.yaml
        metrics.json
        final_model.pt
        final_figure.png
        loss_curve.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .grids import make_tf_grid
from .losses import (
    compute_total_loss,
    compute_global_moment_targets,
    compute_cond_f_mean_target,
)
from .metrics import compute_all_metrics
from .models import build_model
from .signals import build_signal, compute_marginals
from .visualize import make_experiment_figure


# --------------------------------------------------------------------------- #
# Cohen-class reference (pseudo-WVD)                                          #
# --------------------------------------------------------------------------- #
def compute_pWVD(
    x: torch.Tensor,
    N: int,
    device: str | torch.device,
    kernel_size: int = 15,
    sigma_k: float = 2.0,
) -> torch.Tensor:
    """
    Smoothed pseudo-Wigner-Ville Distribution (DC-centred).

    Parameters
    ----------
    x           : Tensor (N,)  complex or real signal (cast to complex internally)
    N           : signal length
    device      : torch device
    kernel_size : Gaussian smoothing kernel size (must be odd)
    sigma_k     : smoothing width in pixels

    Returns
    -------
    pWVD : Tensor (N, N)  normalised non-negative TFD (sums to 1)
    """
    x_c = x.to(torch.complex64).to(device)
    x_padded = torch.zeros(2 * N, dtype=torch.complex64, device=device)
    x_padded[:N] = x_c

    half = N // 2
    tau_range = torch.arange(-half, half, device=device)

    WVD = torch.zeros(N, N, device=device)
    for t in range(N):
        t_plus = (t + tau_range) % (2 * N)
        t_minus = (t - tau_range) % (2 * N)
        lag_product = x_padded[t_plus] * x_padded[t_minus].conj()
        row = torch.fft.fft(lag_product, norm="ortho")
        WVD[t, :] = row.real

    # DC-centre the frequency axis
    WVD = torch.fft.fftshift(WVD, dim=1)

    # 2-D Gaussian smoothing (Cohen-class operation)
    centre = kernel_size // 2
    idx = torch.arange(kernel_size, dtype=torch.float32, device=device)
    g1d = torch.exp(-0.5 * ((idx - centre) / sigma_k) ** 2)
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()

    kernel = g2d.unsqueeze(0).unsqueeze(0)
    pWVD = WVD.unsqueeze(0).unsqueeze(0)
    pWVD = F.conv2d(pWVD, kernel, padding=kernel_size // 2)
    pWVD = pWVD.squeeze(0).squeeze(0)

    pWVD = torch.clamp(pWVD, min=0.0)
    pWVD = pWVD / (pWVD.sum() + 1e-12)

    assert pWVD.min() >= 0.0, "pWVD has negative values after clamp"
    assert abs(pWVD.sum().item() - 1.0) < 1e-4, \
        f"pWVD does not sum to 1 (got {pWVD.sum().item():.6f})"

    return pWVD


# --------------------------------------------------------------------------- #
# Single-run training                                                          #
# --------------------------------------------------------------------------- #
def run_experiment(config: dict) -> dict[str, Any]:
    """
    Run one training experiment from a config dict.

    Parameters
    ----------
    config : dict  (see configs/ for examples)

    Returns
    -------
    dict with keys 'metrics', 'output_dir'
    """
    # ------------------------------------------------------------------ setup
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = config.get("device", None)
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    device = torch.device(device)

    N = int(config.get("N", 128))

    # ---------------------------------------------------------------- signal
    signal_cfg = config.get("signal", {})
    sig_name = signal_cfg.get("name", "real_linear_chirp")
    sig_params = deepcopy(signal_cfg.get("params", {}))
    sig_params.setdefault("N", N)

    t_grid, x, sig_meta = build_signal(sig_name, sig_params)
    time_energy, freq_energy = compute_marginals(x)

    time_energy = time_energy.to(device)
    freq_energy = freq_energy.to(device)

    # ------------------------------------------------------------------ grid
    grid = make_tf_grid(N, device=device)
    coords = grid["coords"]
    TT = grid["TT"]
    FF = grid["FF"]
    f_norm = grid["f_norm"]

    # --------------------------------------------------------- reference pWVD
    pWVD = compute_pWVD(x, N, device)

    # ---------------------------------------------------------- loss targets
    lambdas = config.get("lambdas", {"marginal": 1.0})
    loss_targets: dict[str, Any] = {}

    if lambdas.get("global_moment", 0.0) > 0.0:
        loss_targets["global_moment_targets"] = compute_global_moment_targets(
            pWVD, TT, FF
        )

    if lambdas.get("cond_f_mean", 0.0) > 0.0:
        # freq scale: t_grid ∈ [-0.5, 0.5), f_norm ∈ [-1, 1] → scale = 2
        cond_target = compute_cond_f_mean_target(
            t_grid, sig_meta, f_scale=2.0
        ).to(device)
        loss_targets["cond_f_mean_target"] = cond_target

    if lambdas.get("cond_f_spread", 0.0) > 0.0:
        # Default: zero spread target (narrow ridge)
        loss_targets["cond_f_spread_target"] = torch.zeros(N, device=device)

    # ----------------------------------------------------------------- model
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "TFD_Network")
    model_params = model_cfg.get("params", {})
    net = build_model(model_name, model_params).to(device)

    # --------------------------------------------------------------- training
    train_cfg = config.get("training", {})
    num_epochs = int(train_cfg.get("num_epochs", 2000))
    lr = float(train_cfg.get("lr", 1e-4))
    log_every = int(train_cfg.get("log_every", 200))
    record_every = int(train_cfg.get("record_every", 10))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_history: dict[str, list[float]] = {}
    loss_rows: list[dict] = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred_flat = net(coords)
        P = pred_flat.view(N, N)
        P = P / (P.sum() + 1e-8)

        loss_dict = compute_total_loss(
            P, time_energy, freq_energy, TT, FF, lambdas, loss_targets
        )
        loss_dict["total"].backward()
        optimizer.step()

        assert torch.isfinite(P).all(), f"NaN/Inf in P at epoch {epoch}"
        assert torch.isfinite(loss_dict["total"]), \
            f"NaN/Inf in loss at epoch {epoch}"

        if epoch % record_every == 0:
            row = {"epoch": epoch}
            for k, v in loss_dict.items():
                val = float(v.item())
                row[k] = val
                if k not in loss_history:
                    loss_history[k] = []
                loss_history[k].append(val)
            loss_rows.append(row)

        if epoch % log_every == 0:
            with torch.no_grad():
                pred_time = P.sum(dim=1)
                pred_freq = P.sum(dim=0)
                rel_t = ((pred_time - time_energy).norm() /
                         (time_energy.norm() + 1e-12)).item()
                rel_f = ((pred_freq - freq_energy).norm() /
                         (freq_energy.norm() + 1e-12)).item()
                print(
                    f"Epoch {epoch:5d} | "
                    f"loss={loss_dict['total'].item():.3e} | "
                    f"rel_t={rel_t:.4f} | rel_f={rel_f:.4f}"
                )

    # ---------------------------------------------------------- final metrics
    with torch.no_grad():
        pred_flat = net(coords)
        P_final = pred_flat.view(N, N)
        P_final = P_final / (P_final.sum() + 1e-8)

    final_metrics = compute_all_metrics(P_final, time_energy, freq_energy, x)
    final_metrics["seed"] = seed
    final_metrics["signal"] = sig_name
    final_metrics["num_epochs"] = num_epochs

    # ------------------------------------------------------------ output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition = config.get("condition_name", "run")
    out_dir = Path(config.get("output_root", "results")) / \
        f"{timestamp}_{sig_name}_{condition}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import yaml
    with open(out_dir / "config.yaml", "w") as f:
        # Serialize: remove non-serialisable items (e.g. lambdas) safely
        yaml.dump(_make_serializable(config), f, default_flow_style=False)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save model
    torch.save(net.state_dict(), out_dir / "final_model.pt")

    # Save loss curve
    if loss_rows:
        keys = list(loss_rows[0].keys())
        with open(out_dir / "loss_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(loss_rows)

    # Save figure
    with torch.no_grad():
        pred_time_np = P_final.sum(dim=1)
        pred_freq_np = P_final.sum(dim=0)
        rel_err_t = final_metrics["rel_err_t"]
        rel_err_f = final_metrics["rel_err_f"]

    fig = make_experiment_figure(
        t_grid.cpu(), f_norm.cpu(),
        time_energy.cpu(), freq_energy.cpu(),
        pred_time_np.cpu(), pred_freq_np.cpu(),
        P_final.cpu(), pWVD.cpu(),
        rel_err_t, rel_err_f,
        loss_history=loss_history,
        epoch_stride=record_every,
        title_suffix=f" ({condition})",
    )
    fig.savefig(out_dir / "final_figure.png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)

    print(f"\nResults saved to: {out_dir}")
    print(f"  rel_err_t = {final_metrics['rel_err_t']:.6f}")
    print(f"  rel_err_f = {final_metrics['rel_err_f']:.6f}")
    print(f"  symmetry_err = {final_metrics['symmetry_err']:.6e}")

    return {"metrics": final_metrics, "output_dir": str(out_dir)}


# --------------------------------------------------------------------------- #
# Multi-seed runner                                                            #
# --------------------------------------------------------------------------- #
def run_experiment_multi_seed(
    config: dict,
    seeds: list[int],
    parallel: bool = False,
    n_jobs: int = -1,
) -> list[dict]:
    """
    Run the same experiment with multiple seeds.

    Parameters
    ----------
    config   : base config dict (seed field is overridden per run)
    seeds    : list of integer seeds
    parallel : if True, use multiprocessing.Pool for CPU-parallel execution
    n_jobs   : number of worker processes (-1 = number of CPU cores)

    Returns
    -------
    list of result dicts, one per seed
    """
    configs = []
    for s in seeds:
        cfg = deepcopy(config)
        cfg["seed"] = s
        configs.append(cfg)

    if parallel:
        import multiprocessing
        workers = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(run_experiment, configs)
    else:
        results = [run_experiment(cfg) for cfg in configs]

    return results


# --------------------------------------------------------------------------- #
# Serialisation helper                                                         #
# --------------------------------------------------------------------------- #
def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if callable(obj):
        return "<callable>"
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
