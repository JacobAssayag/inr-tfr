"""
signals.py
==========
Real-valued signal generators for time-frequency INR experiments.

All generators return:
    t_grid   : Tensor (N,)   endpoint-safe physical time axis in [-0.5, 0.5)
    x        : Tensor (N,)   real-valued signal (float32)
    metadata : dict          true component centers or chirp ridge info

Endpoint-safe sampling uses  t = arange(N) / N - 0.5  so the grid spans
[-0.5, 0.5 - 1/N] and is compatible with a periodic (DFT) interpretation.

Frequency marginals are computed with fftfreq + fftshift so that DC is
centred at index N//2 and the frequency axis is monotonically increasing.

Parseval consistency is enforced by `compute_marginals`: the relative error
between time-domain energy (∑|x(t)|²) and frequency-domain energy (∑|X(f)|²)
must be below 1e-3, as verified by an assertion in that function.
"""

import torch


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _time_grid(N: int) -> torch.Tensor:
    """Endpoint-safe time axis in [-0.5, 0.5 - 1/N]."""
    return torch.arange(N, dtype=torch.float32) / N - 0.5


def _freq_grid(N: int) -> torch.Tensor:
    """DC-centred frequency axis via fftfreq + fftshift, values in [-0.5, 0.5)."""
    return torch.fft.fftshift(torch.fft.fftfreq(N))


def compute_marginals(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalised time and frequency energy marginals for a real signal.

    Parameters
    ----------
    x : Tensor (N,)   real-valued signal

    Returns
    -------
    time_energy : Tensor (N,)   normalised |x(t)|^2, sums to 1
    freq_energy : Tensor (N,)   normalised |X(f)|^2 (fftshifted, DC centred),
                                sums to 1
    """
    N = x.shape[0]

    raw_time = x.float().pow(2)                              # |x(t)|^2
    X = torch.fft.fft(x.float(), norm="ortho")              # unitary FFT
    raw_freq_unshifted = X.abs().pow(2)                     # |X(f)|^2

    # Parseval check (on raw values)
    parseval_err = (raw_time.sum() - raw_freq_unshifted.sum()).abs() / (
        raw_time.sum() + 1e-12
    )
    assert parseval_err < 1e-3, f"Parseval violation: {parseval_err:.2e}"

    # DC-centre the frequency energy so it matches _freq_grid ordering
    raw_freq = torch.fft.fftshift(raw_freq_unshifted)

    time_energy = raw_time / raw_time.sum()
    freq_energy = raw_freq / raw_freq.sum()

    assert (time_energy.sum() - 1.0).abs() < 1e-5, "time_energy does not sum to 1"
    assert (freq_energy.sum() - 1.0).abs() < 1e-5, "freq_energy does not sum to 1"

    return time_energy, freq_energy


def _validate_signal(t_grid: torch.Tensor, x: torch.Tensor) -> None:
    """Assert real dtype, finite values, and matching lengths."""
    assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
    assert torch.isfinite(x).all(), "Signal contains NaN/Inf"
    assert t_grid.shape == x.shape, "t_grid and x shape mismatch"


# --------------------------------------------------------------------------- #
# Public signal generators                                                     #
# --------------------------------------------------------------------------- #
def real_gabor(
    N: int = 128,
    f0: float = 10.0,
    sigma: float = 0.1,
    t0: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Single real-valued Gabor atom.

        x(t) = cos(2π f0 (t - t0)) · exp(-(t - t0)² / (2 σ²))

    Parameters
    ----------
    N     : number of samples
    f0    : centre frequency (cycles per unit time, physical units)
    sigma : Gaussian width (time units)
    t0    : centre time

    Returns
    -------
    t_grid, x, metadata
    """
    t_grid = _time_grid(N)
    tau = t_grid - t0
    x = torch.cos(2.0 * torch.pi * f0 * tau) * torch.exp(-tau**2 / (2 * sigma**2))
    x = x.float()

    _validate_signal(t_grid, x)

    metadata = {
        "type": "real_gabor",
        "N": N,
        "f0": f0,
        "sigma": sigma,
        "t0": t0,
        "true_center_t": t0,
        "true_center_f": f0,          # positive lobe of the two-sided spectrum
    }
    return t_grid, x, metadata


def two_real_gabors(
    N: int = 128,
    f0: float = 8.0,
    f1: float = 20.0,
    sigma: float = 0.08,
    t0: float = -0.15,
    t1: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Sum of two real-valued Gabor atoms.

    Returns
    -------
    t_grid, x, metadata
        metadata['component_centers'] = [(t0, f0), (t1, f1)]
    """
    t_grid = _time_grid(N)

    def _gabor(tau, f):
        return torch.cos(2.0 * torch.pi * f * tau) * torch.exp(
            -tau**2 / (2 * sigma**2)
        )

    g0 = _gabor(t_grid - t0, f0)
    g1 = _gabor(t_grid - t1, f1)
    x = (g0 + g1).float()

    _validate_signal(t_grid, x)

    metadata = {
        "type": "two_real_gabors",
        "N": N,
        "f0": f0,
        "f1": f1,
        "sigma": sigma,
        "t0": t0,
        "t1": t1,
        "component_centers": [(t0, f0), (t1, f1)],
    }
    return t_grid, x, metadata


def real_linear_chirp(
    N: int = 128,
    k: float = 40.0,
    sigma: float = 0.12,
    t0: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Real-valued windowed linear (constant-rate) chirp.

        x(t) = cos(π k (t - t0)²) · exp(-(t - t0)² / (2 σ²))

    Instantaneous frequency:  f_inst(t) = k · (t - t0)  [cycles per unit time]

    Parameters
    ----------
    N     : number of samples
    k     : chirp rate (Hz/s in normalised units)
    sigma : Gaussian envelope width
    t0    : time origin of the chirp

    Returns
    -------
    t_grid, x, metadata
        metadata['true_chirp_ridge'] callable: t → f_inst(t)
    """
    t_grid = _time_grid(N)
    tau = t_grid - t0
    x = (torch.cos(torch.pi * k * tau**2) * torch.exp(-tau**2 / (2 * sigma**2))).float()

    _validate_signal(t_grid, x)

    metadata = {
        "type": "real_linear_chirp",
        "N": N,
        "k": k,
        "sigma": sigma,
        "t0": t0,
        "true_chirp_ridge": lambda t: float(k) * (t - float(t0)),
    }
    return t_grid, x, metadata


def multi_real_chirp(
    N: int = 128,
    chirp_params: list[dict] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Sum of multiple real-valued linear chirps.

    Parameters
    ----------
    N            : number of samples
    chirp_params : list of dicts, each with keys 'k', 'sigma', 't0', 'amplitude'.
                   Defaults to two crossing chirps.

    Returns
    -------
    t_grid, x, metadata
        metadata['chirp_ridges'] = list of callables t → f_inst(t)
    """
    if chirp_params is None:
        chirp_params = [
            {"k": 30.0, "sigma": 0.18, "t0": 0.0, "amplitude": 1.0},
            {"k": -20.0, "sigma": 0.18, "t0": 0.0, "amplitude": 0.8},
        ]

    t_grid = _time_grid(N)
    x = torch.zeros(N, dtype=torch.float32)
    ridges = []

    for p in chirp_params:
        k = float(p.get("k", 30.0))
        sigma = float(p.get("sigma", 0.15))
        t0 = float(p.get("t0", 0.0))
        amp = float(p.get("amplitude", 1.0))
        tau = t_grid - t0
        component = amp * torch.cos(torch.pi * k * tau**2) * torch.exp(
            -tau**2 / (2 * sigma**2)
        )
        x = x + component.float()
        ridges.append(lambda t, _k=k, _t0=t0: _k * (t - _t0))

    _validate_signal(t_grid, x)

    metadata = {
        "type": "multi_real_chirp",
        "N": N,
        "chirp_params": chirp_params,
        "chirp_ridges": ridges,
    }
    return t_grid, x, metadata


# --------------------------------------------------------------------------- #
# Registry for config-driven construction                                      #
# --------------------------------------------------------------------------- #
SIGNAL_REGISTRY = {
    "real_gabor": real_gabor,
    "two_real_gabors": two_real_gabors,
    "real_linear_chirp": real_linear_chirp,
    "multi_real_chirp": multi_real_chirp,
}


def build_signal(name: str, params: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a signal by name with a parameter dictionary."""
    if name not in SIGNAL_REGISTRY:
        raise ValueError(
            f"Unknown signal '{name}'. Available: {list(SIGNAL_REGISTRY.keys())}"
        )
    return SIGNAL_REGISTRY[name](**params)
