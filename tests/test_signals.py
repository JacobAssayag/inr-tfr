"""
tests/test_signals.py
=====================
Tests for inr_tfr.signals module.

Covers:
- Real-valued dtype
- Finite values
- Parseval consistency
- Marginal normalisation
- Metadata presence
"""

import pytest
import torch

from inr_tfr.signals import (
    real_gabor,
    two_real_gabors,
    real_linear_chirp,
    multi_real_chirp,
    compute_marginals,
    _time_grid,
    _freq_grid,
)

GENERATORS = [
    ("real_gabor", real_gabor, {}),
    ("two_real_gabors", two_real_gabors, {}),
    ("real_linear_chirp", real_linear_chirp, {}),
    ("multi_real_chirp", multi_real_chirp, {}),
]

N = 64  # small N for fast tests


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _call(fn, extra_params):
    return fn(N=N, **extra_params)


# --------------------------------------------------------------------------- #
# 1. Real-valued dtype                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,params", GENERATORS)
def test_real_valued_dtype(name, fn, params):
    _, x, _ = _call(fn, params)
    assert x.dtype == torch.float32, \
        f"{name}: expected float32, got {x.dtype}"
    assert not x.is_complex(), f"{name}: signal must not be complex"


# --------------------------------------------------------------------------- #
# 2. Finite values                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,params", GENERATORS)
def test_finite_values(name, fn, params):
    t_grid, x, _ = _call(fn, params)
    assert torch.isfinite(x).all(), f"{name}: signal contains NaN/Inf"
    assert torch.isfinite(t_grid).all(), f"{name}: t_grid contains NaN/Inf"


# --------------------------------------------------------------------------- #
# 3. Parseval consistency                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,params", GENERATORS)
def test_parseval_consistency(name, fn, params):
    _, x, _ = _call(fn, params)
    raw_time = x.float().pow(2).sum()
    X = torch.fft.fft(x.float(), norm="ortho")
    raw_freq = X.abs().pow(2).sum()
    err = ((raw_time - raw_freq).abs() / (raw_time + 1e-12)).item()
    assert err < 1e-3, f"{name}: Parseval error {err:.2e} exceeds 1e-3"


# --------------------------------------------------------------------------- #
# 4. Marginal normalisation                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,params", GENERATORS)
def test_marginal_normalisation(name, fn, params):
    _, x, _ = _call(fn, params)
    time_energy, freq_energy = compute_marginals(x)
    assert abs(time_energy.sum().item() - 1.0) < 1e-5, \
        f"{name}: time_energy does not sum to 1"
    assert abs(freq_energy.sum().item() - 1.0) < 1e-5, \
        f"{name}: freq_energy does not sum to 1"
    assert time_energy.min().item() >= 0.0, \
        f"{name}: time_energy has negative values"
    assert freq_energy.min().item() >= 0.0, \
        f"{name}: freq_energy has negative values"


# --------------------------------------------------------------------------- #
# 5. Metadata correctness                                                      #
# --------------------------------------------------------------------------- #
def test_gabor_metadata():
    _, _, meta = real_gabor(N=N, f0=12.0, t0=0.1)
    assert meta["true_center_f"] == pytest.approx(12.0)
    assert meta["true_center_t"] == pytest.approx(0.1)
    assert meta["type"] == "real_gabor"


def test_two_gabors_metadata():
    _, _, meta = two_real_gabors(N=N)
    centers = meta["component_centers"]
    assert len(centers) == 2, "Expected 2 component centers"
    for (t_c, f_c) in centers:
        assert isinstance(t_c, float)
        assert isinstance(f_c, float)


def test_chirp_metadata():
    _, _, meta = real_linear_chirp(N=N, k=30.0, t0=0.0)
    ridge = meta["true_chirp_ridge"]
    assert callable(ridge), "true_chirp_ridge must be callable"
    # f_inst(0) = 0, f_inst(1) = 30
    assert ridge(0.0) == pytest.approx(0.0)
    assert ridge(1.0) == pytest.approx(30.0)


def test_multi_chirp_metadata():
    _, _, meta = multi_real_chirp(N=N)
    assert "chirp_ridges" in meta
    assert len(meta["chirp_ridges"]) == len(meta["chirp_params"])


# --------------------------------------------------------------------------- #
# 6. Endpoint-safe time grid                                                   #
# --------------------------------------------------------------------------- #
def test_time_grid_endpoint_safe():
    t = _time_grid(128)
    # Should NOT include 0.5 (endpoint-safe)
    assert t.min().item() == pytest.approx(-0.5, abs=1e-6)
    assert t.max().item() < 0.5


def test_freq_grid_dc_centred():
    f = _freq_grid(128)
    # DC should be at the centre index
    centre_idx = 128 // 2
    assert abs(f[centre_idx].item()) < 1e-6, \
        f"DC not at centre index: f[{centre_idx}] = {f[centre_idx].item()}"


# --------------------------------------------------------------------------- #
# 7. Signal shape                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,fn,params", GENERATORS)
def test_signal_shape(name, fn, params):
    t_grid, x, _ = _call(fn, params)
    assert t_grid.shape == (N,), f"{name}: t_grid shape {t_grid.shape}"
    assert x.shape == (N,), f"{name}: x shape {x.shape}"
