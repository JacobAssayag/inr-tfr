"""
grids.py
========
Coordinate grid utilities for 2-D time-frequency INR.

The INR takes normalised (t_norm, f_norm) ∈ [-1, 1]² as input.
Both axes have N points, matching the N-point signal and FFT.
"""

import torch


def make_tf_grid(N: int, device: str | torch.device = "cpu") -> dict:
    """
    Build the full N×N coordinate grid for the INR.

    Normalised axes run from -1 to 1 (inclusive) with N equally-spaced points.
    The grid has the same number of points as the physical time/frequency axes,
    so marginals computed from P[t_idx, f_idx] align one-to-one with the
    physical marginal vectors (time_energy, freq_energy).

    Parameters
    ----------
    N      : grid resolution (same as signal length)
    device : PyTorch device

    Returns
    -------
    dict with keys:
        t_norm  : Tensor (N,)   normalised time axis in [-1, 1]
        f_norm  : Tensor (N,)   normalised freq axis in [-1, 1]
        TT      : Tensor (N, N) time coordinate meshgrid
        FF      : Tensor (N, N) freq coordinate meshgrid
        coords  : Tensor (N*N, 2) flattened (t, f) pairs for INR input
    """
    t_norm = torch.linspace(-1.0, 1.0, N, device=device)
    f_norm = torch.linspace(-1.0, 1.0, N, device=device)

    TT, FF = torch.meshgrid(t_norm, f_norm, indexing="ij")  # (N, N) each
    coords = torch.stack([TT.reshape(-1), FF.reshape(-1)], dim=-1)  # (N*N, 2)

    # Sanity checks
    assert TT.shape == (N, N), f"TT shape mismatch: {TT.shape}"
    assert FF.shape == (N, N), f"FF shape mismatch: {FF.shape}"
    assert coords.shape == (N * N, 2), f"coords shape mismatch: {coords.shape}"
    assert abs(coords[:, 0].min().item() + 1.0) < 1e-6, "t_norm out of [-1, 1]"
    assert abs(coords[:, 1].max().item() - 1.0) < 1e-6, "f_norm out of [-1, 1]"

    return {
        "t_norm": t_norm,
        "f_norm": f_norm,
        "TT": TT,
        "FF": FF,
        "coords": coords,
    }
