"""
tests/test_losses.py
====================
Tests for inr_tfr.losses module.

Covers:
- Marginal loss is 0 when predicted marginals equal true
- Global moment loss is 0 when moments match
- Symmetry loss is 0 for a symmetric P
- Conditional |f| mean loss is non-negative
"""

import pytest
import torch

from inr_tfr.grids import make_tf_grid
from inr_tfr.losses import (
    marginal_loss,
    global_moment_loss,
    symmetry_loss,
    conditional_f_mean_loss,
    compute_total_loss,
    compute_global_moment_targets,
)
from inr_tfr.signals import real_gabor, compute_marginals


N = 32


@pytest.fixture
def grid():
    return make_tf_grid(N)


@pytest.fixture
def marginals():
    _, x, _ = real_gabor(N=N)
    te, fe = compute_marginals(x)
    return te, fe


# --------------------------------------------------------------------------- #
# 1. Marginal loss = 0 when predictions are perfect                           #
# --------------------------------------------------------------------------- #
def test_marginal_loss_zero(grid, marginals):
    te, fe = marginals
    # Construct product distribution whose marginals exactly match
    P = torch.outer(te, fe)
    P = P / P.sum()
    result = marginal_loss(P, te, fe)
    assert result["total"].item() < 1e-10, \
        f"Marginal loss should be 0 for exact match: {result['total'].item()}"


# --------------------------------------------------------------------------- #
# 2. Marginal loss > 0 when predictions differ                                #
# --------------------------------------------------------------------------- #
def test_marginal_loss_nonzero(grid, marginals):
    te, fe = marginals
    P = torch.ones(N, N) / (N * N)   # uniform: wrong marginals
    result = marginal_loss(P, te, fe)
    assert result["total"].item() > 0.0


# --------------------------------------------------------------------------- #
# 3. Symmetry loss = 0 for symmetric distribution                             #
# --------------------------------------------------------------------------- #
def test_symmetry_loss_zero():
    f_vec = torch.linspace(-1, 1, N)
    t_vec = torch.linspace(-1, 1, N)
    TT, FF = torch.meshgrid(t_vec, f_vec, indexing="ij")
    P = torch.exp(-TT**2 / 0.5) * torch.exp(-FF**2 / 0.5)
    P = P / P.sum()
    result = symmetry_loss(P)
    assert result["total"].item() < 1e-10, \
        f"Symmetry loss should be 0 for symmetric P: {result['total'].item()}"


# --------------------------------------------------------------------------- #
# 4. Symmetry loss > 0 for asymmetric distribution                            #
# --------------------------------------------------------------------------- #
def test_symmetry_loss_nonzero():
    P = torch.zeros(N, N)
    P[:, N // 2:] = 1.0  # energy only at positive f
    P = P / P.sum()
    result = symmetry_loss(P)
    assert result["total"].item() > 0.0


# --------------------------------------------------------------------------- #
# 5. Global moment loss = 0 when targets are matched                          #
# --------------------------------------------------------------------------- #
def test_global_moment_loss_zero(grid):
    TT = grid["TT"]
    FF = grid["FF"]
    # Construct a simple distribution
    P = torch.exp(-(TT**2 + FF**2) / 0.3)
    P = P / P.sum()
    targets = compute_global_moment_targets(P, TT, FF)
    result = global_moment_loss(
        P, TT, FF,
        targets["mean_t"], targets["mean_absf"],
        targets["var_t"], targets["var_absef"],
    )
    assert result["total"].item() < 1e-10, \
        f"Global moment loss should be 0 for self-reference: {result['total'].item()}"


# --------------------------------------------------------------------------- #
# 6. Conditional |f| mean loss is non-negative                                #
# --------------------------------------------------------------------------- #
def test_cond_f_mean_loss_nonneg(grid):
    FF = grid["FF"]
    P = torch.ones(N, N) / (N * N)
    target = torch.zeros(N)
    result = conditional_f_mean_loss(P, FF, target)
    assert result["total"].item() >= 0.0


# --------------------------------------------------------------------------- #
# 7. compute_total_loss returns all expected keys                              #
# --------------------------------------------------------------------------- #
def test_total_loss_keys(grid, marginals):
    te, fe = marginals
    TT = grid["TT"]
    FF = grid["FF"]
    P = torch.outer(te, fe)
    P = P / P.sum()
    lambdas = {"marginal": 1.0, "symmetry": 0.1}
    result = compute_total_loss(P, te, fe, TT, FF, lambdas, {})
    assert "total" in result
    assert "marginal_time" in result
    assert "marginal_freq" in result
    assert "symmetry" in result
