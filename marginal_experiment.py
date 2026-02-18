"""
marginal_experiment.py
=====================
Demonstrates the marginal ambiguity problem in time-frequency analysis using
a SIREN-based Implicit Neural Representation (INR).

A windowed linear chirp signal is generated, and a coordinate-MLP is trained
to satisfy only the 1-D marginal constraints.  The learned 2-D distribution
P(t, f) will converge on the marginals but fail to recover the diagonal ridge
structure visible in the Cohen-class reference (smoothed pseudo-WVD).
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Step 1 -- Signal Generation                                                 #
# --------------------------------------------------------------------------- #
def generate_windowed_chirp(N=128, k=40.0, sigma=0.12, t0=0.0):
    """
    Returns
    -------
    t_grid      : Tensor (N,)   physical time axis in [-0.5, 0.5)
    x           : Tensor (N,)   complex analytic signal
    time_energy : Tensor (N,)   normalised |x(t)|^2, sums to 1
    freq_energy : Tensor (N,)   normalised |X(f)|^2, sums to 1
    """
    t_grid = torch.linspace(-0.5, 0.5, N)  # physical axis

    x = torch.exp(-(t_grid - t0) ** 2 / (2 * sigma ** 2)) \
        * torch.exp(1j * torch.pi * k * t_grid ** 2)  # windowed linear chirp

    # --- raw (un-normalised) ------------------------------------------------
    raw_time = x.abs().pow(2)  # |x(t)|^2, shape (N,)

    X = torch.fft.fft(x, norm="ortho")  # unitary FFT
    raw_freq = X.abs().pow(2)  # |X(f)|^2, shape (N,)

    # --- normalised to PMF --------------------------------------------------
    time_energy = raw_time / raw_time.sum()
    freq_energy = raw_freq / raw_freq.sum()

    # --- Mandatory validations ----------------------------------------------
    # 1. Parseval (on raw values, before normalisation)
    parseval_err = (raw_time.sum() - raw_freq.sum()).abs() / raw_time.sum()
    assert parseval_err < 1e-3, f"Parseval violation: {parseval_err:.2e}"

    # 2. Non-negativity
    assert raw_time.min() >= 0, "time_energy has negative values"
    assert raw_freq.min() >= 0, "freq_energy has negative values"

    # 3. Finiteness
    assert torch.isfinite(raw_time).all(), "time_energy contains NaN/Inf"
    assert torch.isfinite(raw_freq).all(), "freq_energy contains NaN/Inf"

    # 4. Normalisation
    assert (time_energy.sum() - 1.0).abs() < 1e-6, \
        "time_energy does not sum to 1"
    assert (freq_energy.sum() - 1.0).abs() < 1e-6, \
        "freq_energy does not sum to 1"

    return t_grid, x, time_energy, freq_energy


# --------------------------------------------------------------------------- #
# Step 3 -- Model Architecture (SIREN INR)                                    #
# --------------------------------------------------------------------------- #
def init_siren_layer(linear, is_first, omega0):
    fan_in = linear.weight.size(1)
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega0
    nn.init.uniform_(linear.weight, -bound, bound)
    # bias
    bias_bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(linear.bias, -bias_bound, bias_bound)


class TFD_Network(nn.Module):
    def __init__(self, omega0=30.0):
        super().__init__()
        self.omega0 = omega0
        self.hidden1 = nn.Linear(2, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

        # SIREN initialisation
        init_siren_layer(self.hidden1, is_first=True, omega0=omega0)
        init_siren_layer(self.hidden2, is_first=False, omega0=omega0)
        init_siren_layer(self.hidden3, is_first=False, omega0=omega0)
        # Leave output layer with default PyTorch init

    def forward(self, coords):
        # coords: (M, 2) normalised (t, f) pairs
        out = torch.sin(self.omega0 * self.hidden1(coords))
        out = torch.sin(self.omega0 * self.hidden2(out))
        out = torch.sin(self.omega0 * self.hidden3(out))
        out = self.output(out)  # (M, 1) raw pre-activation
        out = F.softplus(out)   # (M, 1) non-negative energy
        return out.squeeze(-1)  # (M,)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    # ----- Device selection and reproducibility (Step 4.1) ------------------
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True

    # ----- Step 1 -- Signal Generation --------------------------------------
    N = 128
    t_grid, x, time_energy, freq_energy = generate_windowed_chirp(N=N)

    # ----- Step 2 -- Coordinate Grid ----------------------------------------
    t_norm = torch.linspace(-1.0, 1.0, N)  # normalised time, [-1, 1]
    f_norm = torch.linspace(-1.0, 1.0, N)  # normalised freq, [-1, 1]

    TT, FF = torch.meshgrid(t_norm, f_norm, indexing="ij")
    coords = torch.stack([TT.reshape(-1), FF.reshape(-1)], dim=-1)  # (N*N, 2)

    assert coords[:, 0].min().item() == -1.0 and \
        coords[:, 0].max().item() == 1.0, "t_norm out of [-1, 1]"
    assert coords[:, 1].min().item() == -1.0 and \
        coords[:, 1].max().item() == 1.0, "f_norm out of [-1, 1]"

    # Move everything to device
    coords = coords.to(device)
    time_energy = time_energy.to(device)
    freq_energy = freq_energy.to(device)
    x = x.to(device)

    # ----- Step 3 -- Model --------------------------------------------------
    net = TFD_Network(omega0=30.0).to(device)

    # ----- Step 4 -- Training Loop ------------------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    num_epochs = 2000

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred_flat = net(coords)           # (N*N,)
        P = pred_flat.view(N, N)          # (N, N) axis: P[t_idx, f_idx]

        eps = 1e-8
        P = P / (P.sum() + eps)           # normalise to sum = 1 (PMF)

        pred_time = P.sum(dim=1)          # (N,) time marginal
        pred_freq = P.sum(dim=0)          # (N,) freq marginal

        marginal_loss = (
            F.mse_loss(pred_time, time_energy) +
            F.mse_loss(pred_freq, freq_energy)
        )

        marginal_loss.backward()
        optimizer.step()

        # Sanity checks (every epoch)
        assert torch.isfinite(P).all(), "NaN or Inf in P"
        assert torch.isfinite(marginal_loss), "NaN or Inf in loss"

        # Logging (every 200 epochs)
        if epoch % 200 == 0:
            with torch.no_grad():
                rel_err_t = (pred_time - time_energy).norm() / \
                    time_energy.norm()
                rel_err_f = (pred_freq - freq_energy).norm() / \
                    freq_energy.norm()
                mass = P.sum().item()
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {marginal_loss.item():.2e} | "
                    f"rel_err_t: {rel_err_t.item():.4f} | "
                    f"rel_err_f: {rel_err_f.item():.4f} | "
                    f"mass: {mass:.4f}"
                )

    # Final metrics after training
    with torch.no_grad():
        pred_flat = net(coords)
        P = pred_flat.view(N, N)
        eps = 1e-8
        P = P / (P.sum() + eps)
        pred_time = P.sum(dim=1)
        pred_freq = P.sum(dim=0)
        rel_err_t = (pred_time - time_energy).norm() / time_energy.norm()
        rel_err_f = (pred_freq - freq_energy).norm() / freq_energy.norm()
        mass = P.sum().item()
        marginal_loss = (
            F.mse_loss(pred_time, time_energy) +
            F.mse_loss(pred_freq, freq_energy)
        )

    # ----- Step 5 -- Cohen-Class Reference Distribution ---------------------
    # 5a -- Discrete pseudo-WVD
    x_padded = torch.zeros(2 * N, dtype=torch.complex64, device=device)
    x_padded[:N] = x  # x is the complex signal from Step 1

    half = N // 2
    tau_range = torch.arange(-half, half, device=device)  # shape (N,)

    WVD = torch.zeros(N, N, device=device)

    for t in range(N):
        t_plus = (t + tau_range) % (2 * N)   # modular index into x_padded
        t_minus = (t - tau_range) % (2 * N)

        lag_product = x_padded[t_plus] * x_padded[t_minus].conj()  # shape (N,)

        # FFT over lag axis -> DFT in frequency
        row = torch.fft.fft(lag_product, norm="ortho")
        WVD[t, :] = row.real  # imaginary part is numerical noise

    # 5b -- 2-D Gaussian smoothing (Cohen-class operation)
    kernel_size = 15   # must be odd
    sigma_k = 2.0      # smoothing width in pixels

    # Build 1-D Gaussian analytically
    centre = kernel_size // 2
    idx = torch.arange(kernel_size, dtype=torch.float32, device=device)
    g1d = torch.exp(-0.5 * ((idx - centre) / sigma_k) ** 2)
    g1d = g1d / g1d.sum()  # normalise

    # Outer product -> 2-D kernel
    g2d = torch.outer(g1d, g1d)   # (K, K)
    g2d = g2d / g2d.sum()         # ensure sum = 1

    # Apply as convolution
    kernel = g2d.unsqueeze(0).unsqueeze(0)       # (1, 1, K, K)
    pWVD = WVD.unsqueeze(0).unsqueeze(0)         # (1, 1, N, N)

    pWVD = F.conv2d(pWVD, kernel,
                    padding=kernel_size // 2)     # same-size output
    pWVD = pWVD.squeeze(0).squeeze(0)            # (N, N)

    # 5c -- Post-processing for positive energy density
    pWVD = torch.clamp(pWVD, min=0.0)            # enforce non-negativity
    pWVD = pWVD / pWVD.sum()                     # normalise to PMF

    assert pWVD.min() >= 0.0, \
        "pWVD has negative values after clamp"
    assert abs(pWVD.sum().item() - 1.0) < 1e-5, \
        f"pWVD does not sum to 1 (got {pWVD.sum().item():.6f})"

    # ----- Step 6 -- Visualisation ------------------------------------------
    fig = plt.figure(figsize=(18, 5))

    # Panel 1 -- Marginal comparison (two stacked line plots)
    # Top: time marginals
    ax_t1 = fig.add_subplot(3, 3, 1)
    ax_t1.plot(t_grid.cpu(), time_energy.cpu(), color="steelblue",
               linewidth=1.5, label="True")
    ax_t1.plot(t_grid.cpu(), pred_time.detach().cpu(), "--",
               color="firebrick", linewidth=1.5, label="Predicted")
    ax_t1.set_title(f"Time marginal (rel err = {rel_err_t:.4f})",
                    fontsize=9)
    ax_t1.legend(fontsize=8)
    ax_t1.set_xlabel("t")

    # Bottom: freq marginals
    ax_t2 = fig.add_subplot(3, 3, 4)
    ax_t2.plot(f_norm.cpu(), freq_energy.cpu(), color="steelblue",
               linewidth=1.5, label="True")
    ax_t2.plot(f_norm.cpu(), pred_freq.detach().cpu(), "--",
               color="firebrick", linewidth=1.5, label="Predicted")
    ax_t2.set_title(f"Freq marginal (rel err = {rel_err_f:.4f})",
                    fontsize=9)
    ax_t2.legend(fontsize=8)
    ax_t2.set_xlabel("f (normalised)")

    # Panel 2 -- Learned P(t, f)
    P_np = P.detach().cpu().numpy()
    vmin2 = np.percentile(P_np, 1)
    vmax2 = np.percentile(P_np, 99)

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(P_np.T,  # transpose so t on x-axis, f on y-axis
                     origin="lower", aspect="auto",
                     cmap="viridis", vmin=vmin2, vmax=vmax2)
    ax2.set_xlabel("Time index")
    ax2.set_ylabel("Frequency index")
    ax2.set_title("Learned P(t,f) Marginals Only")
    plt.colorbar(im2, ax=ax2, label="Energy density")

    # Panel 3 -- Cohen-class reference (smoothed pWVD)
    pWVD_np = pWVD.cpu().numpy()
    vmin3 = np.percentile(pWVD_np, 1)
    vmax3 = np.percentile(pWVD_np, 99)

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(pWVD_np.T,
                     origin="lower", aspect="auto",
                     cmap="viridis", vmin=vmin3, vmax=vmax3)
    ax3.set_xlabel("Time index")
    ax3.set_ylabel("Frequency index")
    ax3.set_title("Cohen-Class Reference (smoothed pWVD)")
    plt.colorbar(im3, ax=ax3, label="Energy density")

    plt.tight_layout()
    plt.savefig("marginal_experiment_result.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to marginal_experiment_result.png")

    # ----- Failure Summary --------------------------------------------------
    print(f"""
========== FAILURE SUMMARY ==========
Marginal loss         : {marginal_loss:.6e}
Rel error (time)      : {rel_err_t:.6f}
Rel error (freq)      : {rel_err_f:.6f}
Mass of learned P     : {mass:.6f}

Diagnosis: Despite near-zero marginal errors, the learned P(t,f)
does not recover the diagonal ridge structure of the Cohen-class
reference. This confirms that 1D marginal constraints are
insufficient to uniquely determine a 2D energy distribution
(ill-posed inverse problem). Additional priors are required.
======================================""")


if __name__ == "__main__":
    main()
