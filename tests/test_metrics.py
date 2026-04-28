"""
tests/test_metrics.py
=====================
Tests for inr_tfr.metrics module.

Covers:
- Parseval consistency
- Marginal normalisation error
- Symmetry error (perfect symmetric → 0, asymmetric → > 0)
"""

import pytest
import torch
import math

from inr_tfr.metrics import (
    parseval_consistency,
    marginal_rel_error,
    marginal_normalisation_error,
    symmetry_error,
    compute_all_metrics,
)
from inr_tfr.signals import real_gabor, compute_marginals
from inr_tfr.grids import make_tf_grid


N = 64


# --------------------------------------------------------------------------- #
# 1. Parseval consistency                                                      #
# --------------------------------------------------------------------------- #
def test_parseval_gabor():
    _, x, _ = real_gabor(N=N)
    err = parseval_consistency(x)
    assert err < 1e-3, f"Parseval error {err:.2e} too large for Gabor"


def test_parseval_chirp():
    from inr_tfr.signals import real_linear_chirp
    _, x, _ = real_linear_chirp(N=N)
    err = parseval_consistency(x)
    assert err < 1e-3, f"Parseval error {err:.2e} too large for chirp"


# --------------------------------------------------------------------------- #
# 2. Marginal relative error: perfect match → 0                               #
# --------------------------------------------------------------------------- #
def test_marginal_rel_error_perfect():
    _, x, _ = real_gabor(N=N)
    time_energy, freq_energy = compute_marginals(x)
    P = torch.outer(time_energy, freq_energy)          # product distribution
    P = P / P.sum()
    # marginals of a product distribution exactly match the factors
    errs = marginal_rel_error(P, time_energy, freq_energy)
    assert errs["rel_err_t"] < 1e-4
    assert errs["rel_err_f"] < 1e-4


# --------------------------------------------------------------------------- #
# 3. Marginal normalisation                                                    #
# --------------------------------------------------------------------------- #
def test_normalisation_error_zero():
    _, x, _ = real_gabor(N=N)
    te, fe = compute_marginals(x)
    errs = marginal_normalisation_error(te, fe)
    assert errs["time_sum_err"] < 1e-5
    assert errs["freq_sum_err"] < 1e-5


# --------------------------------------------------------------------------- #
# 4. Symmetry error                                                            #
# --------------------------------------------------------------------------- #
def test_symmetry_error_symmetric():
    """A distribution symmetric about the freq axis should give error ≈ 0."""
    N_local = 32
    t_vec = torch.linspace(-1, 1, N_local)
    f_vec = torch.linspace(-1, 1, N_local)
    TT, FF = torch.meshgrid(t_vec, f_vec, indexing="ij")
    # Symmetric Gaussian: P(t,f) = G(t)*G(|f|)
    P = torch.exp(-TT**2 / 0.1) * torch.exp(-FF**2 / 0.1)
    P = P / P.sum()
    err = symmetry_error(P)
    assert err < 1e-10, f"Symmetric distribution has non-zero symmetry error: {err}"


def test_symmetry_error_asymmetric():
    """A distribution with energy only at positive f should give error > 0."""
    N_local = 32
    P = torch.zeros(N_local, N_local)
    P[:, N_local // 2:] = 1.0   # only positive frequencies
    P = P / P.sum()
    err = symmetry_error(P)
    assert err > 0.0, "Asymmetric distribution should have positive symmetry error"


# --------------------------------------------------------------------------- #
# 5. compute_all_metrics                                                       #
# --------------------------------------------------------------------------- #
def test_compute_all_metrics_keys():
    _, x, _ = real_gabor(N=N)
    te, fe = compute_marginals(x)
    P = torch.outer(te, fe)
    P = P / P.sum()
    result = compute_all_metrics(P, te, fe, x)
    for key in ("rel_err_t", "rel_err_f", "parseval_err", "symmetry_err", "mass"):
        assert key in result, f"Missing key: {key}"
    assert result["mass"] == pytest.approx(1.0, abs=1e-5)
