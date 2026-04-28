"""
tests/test_grids.py
===================
Tests for inr_tfr.grids module.

Covers:
- Coordinate grid shape
- Normalised axes in [-1, 1]
- Meshgrid ordering (indexing='ij')
"""

import pytest
import torch

from inr_tfr.grids import make_tf_grid


@pytest.mark.parametrize("N", [32, 64, 128])
def test_grid_shapes(N):
    g = make_tf_grid(N)
    assert g["t_norm"].shape == (N,), f"t_norm shape mismatch for N={N}"
    assert g["f_norm"].shape == (N,), f"f_norm shape mismatch for N={N}"
    assert g["TT"].shape == (N, N), f"TT shape mismatch for N={N}"
    assert g["FF"].shape == (N, N), f"FF shape mismatch for N={N}"
    assert g["coords"].shape == (N * N, 2), f"coords shape mismatch for N={N}"


@pytest.mark.parametrize("N", [32, 64, 128])
def test_axes_range(N):
    g = make_tf_grid(N)
    assert abs(g["t_norm"][0].item() + 1.0) < 1e-6, "t_norm[0] != -1"
    assert abs(g["t_norm"][-1].item() - 1.0) < 1e-6, "t_norm[-1] != 1"
    assert abs(g["f_norm"][0].item() + 1.0) < 1e-6, "f_norm[0] != -1"
    assert abs(g["f_norm"][-1].item() - 1.0) < 1e-6, "f_norm[-1] != 1"


def test_coords_range():
    g = make_tf_grid(64)
    coords = g["coords"]
    assert coords[:, 0].min().item() >= -1.0 - 1e-6
    assert coords[:, 0].max().item() <= 1.0 + 1e-6
    assert coords[:, 1].min().item() >= -1.0 - 1e-6
    assert coords[:, 1].max().item() <= 1.0 + 1e-6


def test_meshgrid_ij_indexing():
    """TT[i, j] should equal t_norm[i] for all j."""
    N = 16
    g = make_tf_grid(N)
    TT = g["TT"]
    t_norm = g["t_norm"]
    for i in range(N):
        assert torch.allclose(TT[i, :], t_norm[i].expand(N)), \
            f"TT[{i},:] does not equal t_norm[{i}]"


def test_coords_two_columns():
    g = make_tf_grid(32)
    coords = g["coords"]
    assert coords.shape[1] == 2, "coords should have 2 columns"


def test_grid_device():
    """Grid should be on the requested device."""
    g = make_tf_grid(16, device="cpu")
    for key, val in g.items():
        assert val.device.type == "cpu", f"{key} is not on cpu"
