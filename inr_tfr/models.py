"""
models.py
=========
SIREN-based Implicit Neural Representation for 2-D time-frequency distributions.

Reference:
  Sitzmann et al., "Implicit Neural Representations with Periodic Activation
  Functions", NeurIPS 2020.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SIREN initialisation                                                         #
# --------------------------------------------------------------------------- #
def init_siren_layer(linear: nn.Linear, is_first: bool, omega0: float) -> None:
    """
    Initialise a linear layer following the SIREN scheme.

    First layer  : U[-1/fan_in,  1/fan_in]
    Hidden layers: U[-√(6/fan_in)/ω₀,  √(6/fan_in)/ω₀]
    """
    fan_in = linear.weight.size(1)
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega0
    nn.init.uniform_(linear.weight, -bound, bound)

    bias_bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(linear.bias, -bias_bound, bias_bound)


# --------------------------------------------------------------------------- #
# TFD_Network                                                                  #
# --------------------------------------------------------------------------- #
class TFD_Network(nn.Module):
    """
    Three-hidden-layer SIREN that maps 2-D coordinates to non-negative energy.

    Architecture
    ------------
    Input  (2,)  →  Linear → sin  →  [hidden × 2]  →  Linear → softplus
    Output (1,)  squeezed to scalar

    Parameters
    ----------
    omega0      : frequency multiplier for SIREN activations
    hidden_dim  : width of each hidden layer (default 256)
    hidden_layers: number of hidden layers (default 3)
    """

    def __init__(
        self,
        omega0: float = 30.0,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.omega0 = omega0

        layers: list[nn.Module] = []
        in_features = 2
        for i in range(hidden_layers):
            linear = nn.Linear(in_features, hidden_dim)
            init_siren_layer(linear, is_first=(i == 0), omega0=omega0)
            layers.append(linear)
            in_features = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, 1)
        # Output layer keeps default PyTorch init

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : Tensor (M, 2)  normalised (t, f) pairs in [-1, 1]²

        Returns
        -------
        Tensor (M,)  non-negative energy density at each coordinate
        """
        out = coords
        for i, layer in enumerate(self.hidden_layers):
            out = torch.sin(self.omega0 * layer(out))
        out = self.output_layer(out)      # (M, 1) raw
        out = F.softplus(out)             # (M, 1) non-negative
        return out.squeeze(-1)            # (M,)


# --------------------------------------------------------------------------- #
# Registry for config-driven construction                                      #
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "TFD_Network": TFD_Network,
}


def build_model(name: str, params: dict) -> nn.Module:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**params)
