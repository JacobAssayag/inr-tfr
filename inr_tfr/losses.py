"""
losses.py
=========
Loss functions for training the time-frequency INR.

All losses operate on a normalised 2-D distribution P (shape N×N, sums to 1)
together with the 1-D marginal targets (time_energy, freq_energy).

Available losses
----------------
marginal_loss          : sum of MSE on time and frequency marginals.
global_moment_loss     : MSE on global first/second moments (uses |f| to
                         avoid signed-frequency cancellation for real signals).
conditional_f_mean_loss: per-time-slice conditional mean of |f|.
conditional_f_spread_loss: per-time-slice conditional spread of |f|.

Losses return a dict so that each component can be logged separately.
The marginal loss is unchanged relative to the original script.
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalise(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise P to sum to 1 in-place-safe fashion."""
    return P / (P.sum() + eps)


# --------------------------------------------------------------------------- #
# 1. Marginal loss (unchanged behaviour)                                       #
# --------------------------------------------------------------------------- #
def marginal_loss(
    P: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    MSE between predicted marginals and target marginals.

    Parameters
    ----------
    P           : Tensor (N, N)  normalised distribution
    time_energy : Tensor (N,)    target time marginal
    freq_energy : Tensor (N,)    target freq marginal (fftshifted, DC centred)

    Returns
    -------
    dict with keys 'marginal_time', 'marginal_freq', 'total'
    """
    pred_time = P.sum(dim=1)   # (N,) sum over frequency axis
    pred_freq = P.sum(dim=0)   # (N,) sum over time axis

    loss_t = F.mse_loss(pred_time, time_energy)
    loss_f = F.mse_loss(pred_freq, freq_energy)

    return {
        "marginal_time": loss_t,
        "marginal_freq": loss_f,
        "total": loss_t + loss_f,
    }


# --------------------------------------------------------------------------- #
# 2. Global moment loss (uses |f| to avoid signed-frequency cancellation)      #
# --------------------------------------------------------------------------- #
def global_moment_loss(
    P: torch.Tensor,
    TT: torch.Tensor,
    FF: torch.Tensor,
    target_mean_t: torch.Tensor,
    target_mean_absf: torch.Tensor,
    target_var_t: torch.Tensor,
    target_var_absef: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    MSE on global first and second moments.

    Uses |f| throughout the frequency dimension to prevent signed-frequency
    cancellation that would occur for real (symmetric) spectra.

    Parameters
    ----------
    P               : Tensor (N, N)  normalised distribution
    TT, FF          : Tensor (N, N)  coordinate meshgrids
    target_mean_t   : scalar target for <t>
    target_mean_absf: scalar target for <|f|>
    target_var_t    : scalar target for var(t)
    target_var_absef: scalar target for var(|f|)

    Returns
    -------
    dict with keys 'mean_t', 'mean_absf', 'var_t', 'var_absef', 'total'
    """
    absF = FF.abs()

    mean_t = (TT * P).sum()
    mean_absf = (absF * P).sum()
    var_t = ((TT - mean_t) ** 2 * P).sum()
    var_absef = ((absF - mean_absf) ** 2 * P).sum()

    l_mean_t = (mean_t - target_mean_t) ** 2
    l_mean_absf = (mean_absf - target_mean_absf) ** 2
    l_var_t = (var_t - target_var_t) ** 2
    l_var_absef = (var_absef - target_var_absef) ** 2

    total = l_mean_t + l_mean_absf + l_var_t + l_var_absef

    return {
        "mean_t": l_mean_t,
        "mean_absf": l_mean_absf,
        "var_t": l_var_t,
        "var_absef": l_var_absef,
        "total": total,
    }


def compute_global_moment_targets(
    P_ref: torch.Tensor,
    TT: torch.Tensor,
    FF: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Extract global moment targets from a reference distribution.

    Parameters
    ----------
    P_ref : Tensor (N, N)  reference distribution (e.g. pWVD), normalised
    TT, FF: Tensor (N, N)  coordinate meshgrids

    Returns
    -------
    dict with keys 'mean_t', 'mean_absf', 'var_t', 'var_absef'
    """
    absF = FF.abs()
    mean_t = (TT * P_ref).sum()
    mean_absf = (absF * P_ref).sum()
    var_t = ((TT - mean_t) ** 2 * P_ref).sum()
    var_absef = ((absF - mean_absf) ** 2 * P_ref).sum()
    return {
        "mean_t": mean_t,
        "mean_absf": mean_absf,
        "var_t": var_t,
        "var_absef": var_absef,
    }


# --------------------------------------------------------------------------- #
# 3. Conditional |f| mean loss                                                 #
# --------------------------------------------------------------------------- #
def conditional_f_mean_loss(
    P: torch.Tensor,
    FF: torch.Tensor,
    target_cond_mean: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Per-time-slice conditional mean of |f|.

        cond_mean(t) = sum_f |f| * P(t,f) / sum_f P(t,f)

    The target is typically derived from the signal's instantaneous frequency
    (absolute value, to match the two-sided spectrum convention).

    Parameters
    ----------
    P               : Tensor (N, N)  normalised distribution
    FF              : Tensor (N, N)  freq coordinate meshgrid
    target_cond_mean: Tensor (N,)    target |f| conditional mean per time slice
    eps             : small stabiliser for division

    Returns
    -------
    dict with keys 'cond_f_mean', 'total'
    """
    absF = FF.abs()
    time_mass = P.sum(dim=1).clamp(min=eps)              # (N,)
    cond_mean = (absF * P).sum(dim=1) / time_mass        # (N,)

    loss_val = F.mse_loss(cond_mean, target_cond_mean)
    return {"cond_f_mean": loss_val, "total": loss_val}


def compute_cond_f_mean_target(
    t_grid: torch.Tensor,
    metadata: dict,
    f_scale: float = 1.0,
) -> torch.Tensor:
    """
    Build target conditional |f| mean from signal metadata.

    For a chirp, uses the true instantaneous frequency |f_inst(t)|.
    For a Gabor, uses |f0| for all t.

    Parameters
    ----------
    t_grid  : Tensor (N,)   physical time grid
    metadata: dict          signal metadata from signals.py
    f_scale : scale factor to map physical freq to normalised f_norm units.
              If t_grid ∈ [-0.5, 0.5) and f_norm ∈ [-1, 1], the mapping is
              f_norm ≈ 2 * f_physical (since fftfreq ∈ [-0.5, 0.5)).

    Returns
    -------
    Tensor (N,)  target |f| per time slice, in normalised units
    """
    sig_type = metadata.get("type", "")

    if sig_type == "real_gabor":
        f0_norm = metadata["f0"] * f_scale
        return torch.full_like(t_grid, abs(f0_norm))

    if sig_type in ("real_linear_chirp", "multi_real_chirp"):
        ridge_fn = metadata.get("true_chirp_ridge") or metadata.get(
            "chirp_ridges", [None]
        )[0]
        if callable(ridge_fn):
            # |f_inst(t)| in physical units, then scale to normalised
            return (
                torch.tensor(
                    [abs(ridge_fn(t.item())) for t in t_grid],
                    dtype=torch.float32,
                )
                * f_scale
            )

    if sig_type == "two_real_gabors":
        # Average of the two component frequencies (crude but non-zero target)
        f0_norm = metadata["f0"] * f_scale
        f1_norm = metadata["f1"] * f_scale
        return torch.full_like(t_grid, (abs(f0_norm) + abs(f1_norm)) / 2.0)

    # Fallback: zero target (will be silently matched if cond loss is active)
    return torch.zeros_like(t_grid)


# --------------------------------------------------------------------------- #
# 4. Conditional |f| spread loss                                               #
# --------------------------------------------------------------------------- #
def conditional_f_spread_loss(
    P: torch.Tensor,
    FF: torch.Tensor,
    target_cond_spread: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Per-time-slice conditional spread (std) of |f|.

        cond_spread(t) = sqrt( sum_f (|f| - cond_mean(t))^2 P(t,f) / sum_f P(t,f) )

    Parameters
    ----------
    P                 : Tensor (N, N)
    FF                : Tensor (N, N)
    target_cond_spread: Tensor (N,)
    eps               : stabiliser

    Returns
    -------
    dict with keys 'cond_f_spread', 'total'
    """
    absF = FF.abs()
    time_mass = P.sum(dim=1).clamp(min=eps)                          # (N,)
    cond_mean = (absF * P).sum(dim=1) / time_mass                   # (N,)
    cond_var = ((absF - cond_mean.unsqueeze(1)) ** 2 * P).sum(dim=1) / time_mass
    cond_spread = cond_var.clamp(min=0.0).sqrt()                    # (N,)

    loss_val = F.mse_loss(cond_spread, target_cond_spread)
    return {"cond_f_spread": loss_val, "total": loss_val}


# --------------------------------------------------------------------------- #
# 5. Symmetry loss (for real signals: P(t,f) = P(t,-f))                       #
# --------------------------------------------------------------------------- #
def symmetry_loss(P: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Enforce Hermitian symmetry of the time-frequency plane for real signals.

    For a real signal the energy distribution must satisfy P(t,f) = P(t,-f).
    With a DC-centred (fftshifted) frequency axis, flipping the freq dimension
    is equivalent to negating f.

    Returns
    -------
    dict with keys 'symmetry', 'total'
    """
    P_flipped = P.flip(dims=[1])   # flip frequency axis (dim=1 is freq)
    loss_val = F.mse_loss(P, P_flipped)
    return {"symmetry": loss_val, "total": loss_val}


# --------------------------------------------------------------------------- #
# Combined loss builder                                                        #
# --------------------------------------------------------------------------- #
def compute_total_loss(
    P: torch.Tensor,
    time_energy: torch.Tensor,
    freq_energy: torch.Tensor,
    TT: torch.Tensor,
    FF: torch.Tensor,
    lambdas: dict[str, float],
    targets: dict,
) -> dict[str, torch.Tensor]:
    """
    Compute a weighted sum of all active loss components.

    Parameters
    ----------
    P           : Tensor (N, N)  normalised distribution
    time_energy : Tensor (N,)    time marginal target
    freq_energy : Tensor (N,)    freq marginal target
    TT, FF      : Tensor (N, N)  coordinate grids
    lambdas     : dict mapping loss name → weight (0.0 disables)
    targets     : dict of pre-computed target values for each loss

    Returns
    -------
    dict with key 'total' and one key per active loss component.
    """
    results: dict[str, torch.Tensor] = {}
    total = torch.zeros(1, device=P.device).squeeze()

    # Marginal loss (always computed for logging, but only added if λ > 0)
    marg = marginal_loss(P, time_energy, freq_energy)
    results["marginal_time"] = marg["marginal_time"]
    results["marginal_freq"] = marg["marginal_freq"]
    lam_m = lambdas.get("marginal", 1.0)
    total = total + lam_m * marg["total"]

    # Global moment loss
    lam_gm = lambdas.get("global_moment", 0.0)
    if lam_gm > 0.0 and "global_moment_targets" in targets:
        gmt = targets["global_moment_targets"]
        gm = global_moment_loss(
            P, TT, FF,
            gmt["mean_t"], gmt["mean_absf"],
            gmt["var_t"], gmt["var_absef"],
        )
        results["gm_mean_t"] = gm["mean_t"]
        results["gm_mean_absf"] = gm["mean_absf"]
        results["gm_var_t"] = gm["var_t"]
        results["gm_var_absef"] = gm["var_absef"]
        total = total + lam_gm * gm["total"]

    # Conditional |f| mean loss
    lam_cm = lambdas.get("cond_f_mean", 0.0)
    if lam_cm > 0.0 and "cond_f_mean_target" in targets:
        cm = conditional_f_mean_loss(P, FF, targets["cond_f_mean_target"])
        results["cond_f_mean"] = cm["cond_f_mean"]
        total = total + lam_cm * cm["total"]

    # Conditional |f| spread loss
    lam_cs = lambdas.get("cond_f_spread", 0.0)
    if lam_cs > 0.0 and "cond_f_spread_target" in targets:
        cs = conditional_f_spread_loss(P, FF, targets["cond_f_spread_target"])
        results["cond_f_spread"] = cs["cond_f_spread"]
        total = total + lam_cs * cs["total"]

    # Symmetry loss
    lam_sym = lambdas.get("symmetry", 0.0)
    if lam_sym > 0.0:
        sym = symmetry_loss(P)
        results["symmetry"] = sym["symmetry"]
        total = total + lam_sym * sym["total"]

    results["total"] = total
    return results
