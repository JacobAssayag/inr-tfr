"""
metrics.py
==========
Evaluation metrics for the time-frequency INR.

Metrics
-------
parseval_consistency   : relative error |∑|x|² - ∑|X|²| / ∑|x|²
marginal_rel_error     : relative L2 error between predicted and true marginals
symmetry_error         : mean squared asymmetry of P(t,f) about f=0
"""

import torch


# --------------------------------------------------------------------------- #
# 1. Parseval consistency                                                      #
# --------------------------------------------------------------------------- #
def parseval_consistency(x: torch.Tensor) -> float:
    """
    Relative error between time-domain and frequency-domain energy.

        err = |∑|x(t)|² - ∑|X(f)|²| / ∑|x(t)|²

    Uses the unitary ('ortho') DFT normalisation so both sums should match.

    Parameters
    ----------
    x : Tensor (N,)  real-valued signal

    Returns
    -------
    float  (should be < 1e-3 for well-formed signals)
    """
    raw_time = x.float().pow(2).sum()
    X = torch.fft.fft(x.float(), norm="ortho")
    raw_freq = X.abs().pow(2).sum()
    return ((raw_time - raw_freq).abs() / (raw_time + 1e-12)).item()


# --------------------------------------------------------------------------- #
# 2. Marginal relative error                                                   #
# --------------------------------------------------------------------------- #
def marginal_rel_error(
    P: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
) -> dict[str, float]:
    """
    Relative L2 error between predicted and true marginals.

    Parameters
    ----------
    P           : Tensor (N, N)  normalised predicted distribution
    time_energy : Tensor (N,)    true time marginal
    freq_energy : Tensor (N,)    true freq marginal

    Returns
    -------
    dict with keys 'rel_err_t', 'rel_err_f'
    """
    pred_time = P.sum(dim=1)
    pred_freq = P.sum(dim=0)

    rel_t = (
        (pred_time - time_energy).norm() / (time_energy.norm() + 1e-12)
    ).item()
    rel_f = (
        (pred_freq - freq_energy).norm() / (freq_energy.norm() + 1e-12)
    ).item()

    return {"rel_err_t": rel_t, "rel_err_f": rel_f}


# --------------------------------------------------------------------------- #
# 3. Marginal normalisation check                                              #
# --------------------------------------------------------------------------- #
def marginal_normalisation_error(
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
) -> dict[str, float]:
    """
    Check that marginals sum to 1.

    Returns
    -------
    dict with keys 'time_sum_err', 'freq_sum_err'
    """
    return {
        "time_sum_err": abs(time_energy.sum().item() - 1.0),
        "freq_sum_err": abs(freq_energy.sum().item() - 1.0),
    }


# --------------------------------------------------------------------------- #
# 4. Symmetry metric                                                           #
# --------------------------------------------------------------------------- #
def symmetry_error(P: torch.Tensor) -> float:
    """
    Mean squared asymmetry of P(t, f) about f = 0.

    For a real signal the TFD should satisfy P(t, f) = P(t, -f).  With a
    DC-centred (fftshifted) frequency axis, flipping the frequency dimension
    maps f → -f.

    Parameters
    ----------
    P : Tensor (N, N)  distribution with shape (time, freq)

    Returns
    -------
    float  (0 = perfectly symmetric, higher = more asymmetric)
    """
    P_flipped = P.flip(dims=[1])
    return torch.mean((P - P_flipped) ** 2).item()


# --------------------------------------------------------------------------- #
# 5. Convenience: compute all metrics at once                                  #
# --------------------------------------------------------------------------- #
def compute_all_metrics(
    P: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
    x: torch.Tensor,
) -> dict[str, float]:
    """
    Compute and return all metrics in a single dict suitable for JSON export.

    Parameters
    ----------
    P           : Tensor (N, N)  normalised predicted distribution
    time_energy : Tensor (N,)    true time marginal
    freq_energy : Tensor (N,)    true freq marginal
    x           : Tensor (N,)    original real-valued signal

    Returns
    -------
    dict[str, float]
    """
    result: dict[str, float] = {}
    result.update(marginal_rel_error(P, time_energy, freq_energy))
    result.update(marginal_normalisation_error(time_energy, freq_energy))
    result["parseval_err"] = parseval_consistency(x)
    result["symmetry_err"] = symmetry_error(P)
    result["mass"] = P.sum().item()
    return result
