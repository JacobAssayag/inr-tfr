"""
visualize.py
============
Visualisation utilities for INR time-frequency experiments.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import torch


# --------------------------------------------------------------------------- #
# Low-level helpers                                                            #
# --------------------------------------------------------------------------- #
def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _pct_clim(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    return np.percentile(arr, lo), np.percentile(arr, hi)


# --------------------------------------------------------------------------- #
# Marginal comparison plot                                                     #
# --------------------------------------------------------------------------- #
def plot_marginals(
    ax_t: plt.Axes,
    ax_f: plt.Axes,
    t_grid: torch.Tensor,
    f_norm: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
    pred_time: torch.Tensor,
    pred_freq: torch.Tensor,
    rel_err_t: float,
    rel_err_f: float,
) -> None:
    """
    Draw true vs predicted marginals on the provided axes.
    """
    t_np = _to_np(t_grid)
    f_np = _to_np(f_norm)

    ax_t.plot(t_np, _to_np(time_energy), color="steelblue",
              linewidth=1.5, label="True")
    ax_t.plot(t_np, _to_np(pred_time), "--", color="firebrick",
              linewidth=1.5, label="Predicted")
    ax_t.set_title(f"Time marginal (rel err = {rel_err_t:.4f})", fontsize=9)
    ax_t.legend(fontsize=8)
    ax_t.set_xlabel("t")

    ax_f.plot(f_np, _to_np(freq_energy), color="steelblue",
              linewidth=1.5, label="True")
    ax_f.plot(f_np, _to_np(pred_freq), "--", color="firebrick",
              linewidth=1.5, label="Predicted")
    ax_f.set_title(f"Freq marginal (rel err = {rel_err_f:.4f})", fontsize=9)
    ax_f.legend(fontsize=8)
    ax_f.set_xlabel("f (normalised)")


# --------------------------------------------------------------------------- #
# TFD heatmap                                                                  #
# --------------------------------------------------------------------------- #
def plot_tfd(
    ax: plt.Axes,
    P: torch.Tensor,
    title: str = "P(t,f)",
    cbar_label: str = "Energy density",
) -> None:
    """
    Plot a 2-D time-frequency distribution.  Transposes P so that time is on
    the x-axis and frequency on the y-axis.
    """
    P_np = _to_np(P)
    vmin, vmax = _pct_clim(P_np)
    im = ax.imshow(
        P_np.T, origin="lower", aspect="auto",
        cmap="viridis", vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Frequency index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)


# --------------------------------------------------------------------------- #
# Loss curve                                                                   #
# --------------------------------------------------------------------------- #
def plot_loss_curve(
    ax: plt.Axes,
    loss_history: dict[str, list[float]],
    epoch_stride: int = 1,
) -> None:
    """
    Semilogy plot of training loss components.

    Parameters
    ----------
    ax           : matplotlib Axes
    loss_history : dict mapping component name → list of scalar values
    epoch_stride : spacing between recorded epochs
    """
    n = max(len(v) for v in loss_history.values())
    epochs_axis = np.arange(n) * epoch_stride

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, values) in enumerate(loss_history.items()):
        ax.semilogy(
            epochs_axis[: len(values)],
            values,
            label=name,
            color=colours[i % len(colours)],
            linewidth=1.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Loss convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# --------------------------------------------------------------------------- #
# Full experiment figure                                                       #
# --------------------------------------------------------------------------- #
def make_experiment_figure(
    t_grid: torch.Tensor,
    f_norm: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
    pred_time: torch.Tensor,
    pred_freq: torch.Tensor,
    P: torch.Tensor,
    pWVD: torch.Tensor,
    rel_err_t: float,
    rel_err_f: float,
    loss_history: dict[str, list[float]] | None = None,
    epoch_stride: int = 1,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Produce the standard 3 (or 4) panel experiment figure.

    Panels:
      1. Time marginal comparison
      2. Freq marginal comparison
      3. Learned P(t, f)
      4. Cohen-class reference pWVD
      [5. Loss curve if loss_history is provided]
    """
    ncols = 3 if loss_history is None else 4
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

    # Top row: marginals
    plot_marginals(
        axes[0, 0], axes[1, 0],
        t_grid, f_norm,
        time_energy, freq_energy,
        pred_time, pred_freq,
        rel_err_t, rel_err_f,
    )

    # Learned P(t,f)
    plot_tfd(axes[0, 1], P, title=f"Learned P(t,f){title_suffix}")
    axes[1, 1].axis("off")  # placeholder below learned TFD

    # Reference pWVD
    plot_tfd(axes[0, 2], pWVD, title="Cohen-Class Reference (smoothed pWVD)")
    axes[1, 2].axis("off")

    if loss_history is not None:
        plot_loss_curve(axes[0, 3], loss_history, epoch_stride=epoch_stride)
        axes[1, 3].axis("off")

    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Multi-run comparison figure                                                  #
# --------------------------------------------------------------------------- #
def make_comparison_figure(
    conditions: list[dict],
    t_grid: torch.Tensor,
    f_norm: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
    pWVD: torch.Tensor,
) -> plt.Figure:
    """
    Compare P(t,f) from multiple experimental conditions side by side.

    Parameters
    ----------
    conditions : list of dicts, each with keys:
                   'P'       (Tensor N×N)
                   'label'   (str)
                   'rel_err_t' (float, optional)
                   'rel_err_f' (float, optional)
    """
    n = len(conditions) + 1  # +1 for reference
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for i, cond in enumerate(conditions):
        P_np = _to_np(cond["P"])
        vmin, vmax = _pct_clim(P_np)
        im = axes[i].imshow(
            P_np.T, origin="lower", aspect="auto",
            cmap="viridis", vmin=vmin, vmax=vmax,
        )
        label = cond.get("label", f"Condition {i+1}")
        rel_t = cond.get("rel_err_t", float("nan"))
        rel_f = cond.get("rel_err_f", float("nan"))
        axes[i].set_title(
            f"{label}\n(err_t={rel_t:.3f}, err_f={rel_f:.3f})", fontsize=9
        )
        axes[i].set_xlabel("Time index")
        axes[i].set_ylabel("Frequency index")
        plt.colorbar(im, ax=axes[i])

    plot_tfd(axes[-1], pWVD, title="Reference (pWVD)")
    plt.tight_layout()
    return fig
