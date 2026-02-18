"""
moment_experiment.py
====================
Follow-up to marginal_experiment.py — adds first and second moment constraints
to the marginal training, testing whether moment matching can resolve the
ambiguity without full 2D supervision.

The marginal operator loses ALL information about time-frequency correlations.
For a chirp with instantaneous frequency f_inst(t) = k*t, the time-frequency
covariance should be non-zero.  Moment constraints provide this missing
correlation information without requiring full 2D supervision.

Moment definitions (discrete, for normalised P with sum(P)=1):
  First moments:
    <t>  = sum_{t,f} t * P(t,f)
    <f>  = sum_{t,f} f * P(t,f)
  Second moments:
    var_t  = sum_{t,f} (t - <t>)^2 * P(t,f)
    var_f  = sum_{t,f} (f - <f>)^2 * P(t,f)
    cov_tf = sum_{t,f} (t - <t>)(f - <f>) * P(t,f)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Step 1 -- Signal Generation (reused from marginal_experiment)               #
# --------------------------------------------------------------------------- #
def generate_windowed_chirp(N=128, k=40.0, sigma=0.12, t0=0.0):
    """
    Returns
    -------
    t_grid      : Tensor (N,)   physical time axis in [-0.5, 0.5)
    x           : Tensor (N,)   complex analytic signal
    time_energy : Tensor (N,)   normalised |x(t)|^2, sums to 1
    freq_energy : Tensor (N,)   normalised |X(f)|^2, sums to 1 (fftshifted,
                                centred so DC is at index N//2)
    """
    t_grid = torch.linspace(-0.5, 0.5, N)

    x = torch.exp(-(t_grid - t0) ** 2 / (2 * sigma ** 2)) \
        * torch.exp(1j * torch.pi * k * t_grid ** 2)

    raw_time = x.abs().pow(2)
    X = torch.fft.fft(x, norm="ortho")
    raw_freq = X.abs().pow(2)

    # Mandatory validations (on raw, before shift/normalisation)
    parseval_err = (raw_time.sum() - raw_freq.sum()).abs() / raw_time.sum()
    assert parseval_err < 1e-3, f"Parseval violation: {parseval_err:.2e}"
    assert raw_time.min() >= 0, "time_energy has negative values"
    assert raw_freq.min() >= 0, "freq_energy has negative values"
    assert torch.isfinite(raw_time).all(), "time_energy contains NaN/Inf"
    assert torch.isfinite(raw_freq).all(), "freq_energy contains NaN/Inf"

    time_energy = raw_time / raw_time.sum()
    # fftshift so that DC is centred — matches the f_norm grid [-1, 1]
    freq_energy = torch.fft.fftshift(raw_freq) / raw_freq.sum()

    assert (time_energy.sum() - 1.0).abs() < 1e-6, \
        "time_energy does not sum to 1"
    assert (freq_energy.sum() - 1.0).abs() < 1e-6, \
        "freq_energy does not sum to 1"

    return t_grid, x, time_energy, freq_energy


# --------------------------------------------------------------------------- #
# Step 3 -- Model Architecture (SIREN INR, reused from marginal_experiment)   #
# --------------------------------------------------------------------------- #
def init_siren_layer(linear, is_first, omega0):
    fan_in = linear.weight.size(1)
    if is_first:
        bound = 1.0 / fan_in
    else:
        bound = math.sqrt(6.0 / fan_in) / omega0
    nn.init.uniform_(linear.weight, -bound, bound)
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

        init_siren_layer(self.hidden1, is_first=True, omega0=omega0)
        init_siren_layer(self.hidden2, is_first=False, omega0=omega0)
        init_siren_layer(self.hidden3, is_first=False, omega0=omega0)

    def forward(self, coords):
        out = torch.sin(self.omega0 * self.hidden1(coords))
        out = torch.sin(self.omega0 * self.hidden2(out))
        out = torch.sin(self.omega0 * self.hidden3(out))
        out = self.output(out)
        out = F.softplus(out)
        return out.squeeze(-1)


# --------------------------------------------------------------------------- #
# Moment computation helpers                                                  #
# --------------------------------------------------------------------------- #
def compute_moments(P, TT, FF):
    """
    Compute first and second moments of a 2-D distribution P(t, f).

    Parameters
    ----------
    P  : Tensor (N, N)  normalised distribution (sums to 1)
    TT : Tensor (N, N)  time coordinate grid  (TT[i,j] = t_norm[i])
    FF : Tensor (N, N)  freq coordinate grid  (FF[i,j] = f_norm[j])

    Returns
    -------
    mean_t  : scalar  <t>
    mean_f  : scalar  <f>
    var_t   : scalar  var(t)
    var_f   : scalar  var(f)
    cov_tf  : scalar  cov(t, f)
    """
    mean_t = (TT * P).sum()
    mean_f = (FF * P).sum()
    var_t = ((TT - mean_t) ** 2 * P).sum()
    var_f = ((FF - mean_f) ** 2 * P).sum()
    cov_tf = ((TT - mean_t) * (FF - mean_f) * P).sum()
    return mean_t, mean_f, var_t, var_f, cov_tf


# --------------------------------------------------------------------------- #
# Cohen-class reference (pseudo-WVD)                                          #
# --------------------------------------------------------------------------- #
def compute_pWVD(x, N, device):
    """Compute smoothed pseudo-Wigner-Ville Distribution (frequency-centred)."""
    x_padded = torch.zeros(2 * N, dtype=torch.complex64, device=device)
    x_padded[:N] = x

    half = N // 2
    tau_range = torch.arange(-half, half, device=device)

    WVD = torch.zeros(N, N, device=device)
    for t in range(N):
        t_plus = (t + tau_range) % (2 * N)
        t_minus = (t - tau_range) % (2 * N)
        lag_product = x_padded[t_plus] * x_padded[t_minus].conj()
        row = torch.fft.fft(lag_product, norm="ortho")
        WVD[t, :] = row.real

    # fftshift the frequency axis so DC is centred — matches f_norm [-1, 1]
    WVD = torch.fft.fftshift(WVD, dim=1)

    # 2-D Gaussian smoothing
    kernel_size = 15
    sigma_k = 2.0
    centre = kernel_size // 2
    idx = torch.arange(kernel_size, dtype=torch.float32, device=device)
    g1d = torch.exp(-0.5 * ((idx - centre) / sigma_k) ** 2)
    g1d = g1d / g1d.sum()
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()

    kernel = g2d.unsqueeze(0).unsqueeze(0)
    pWVD = WVD.unsqueeze(0).unsqueeze(0)
    pWVD = F.conv2d(pWVD, kernel, padding=kernel_size // 2)
    pWVD = pWVD.squeeze(0).squeeze(0)

    # Post-processing
    pWVD = torch.clamp(pWVD, min=0.0)
    pWVD = pWVD / pWVD.sum()

    assert pWVD.min() >= 0.0, "pWVD has negative values after clamp"
    assert abs(pWVD.sum().item() - 1.0) < 1e-5, \
        f"pWVD does not sum to 1 (got {pWVD.sum().item():.6f})"

    return pWVD


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    # ----- Device selection and reproducibility -----------------------------
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
    t_norm = torch.linspace(-1.0, 1.0, N)
    f_norm = torch.linspace(-1.0, 1.0, N)

    TT, FF = torch.meshgrid(t_norm, f_norm, indexing="ij")
    coords = torch.stack([TT.reshape(-1), FF.reshape(-1)], dim=-1)

    assert coords[:, 0].min().item() == -1.0 and \
        coords[:, 0].max().item() == 1.0, "t_norm out of [-1, 1]"
    assert coords[:, 1].min().item() == -1.0 and \
        coords[:, 1].max().item() == 1.0, "f_norm out of [-1, 1]"

    # Move to device
    coords = coords.to(device)
    time_energy = time_energy.to(device)
    freq_energy = freq_energy.to(device)
    x = x.to(device)
    TT = TT.to(device)
    FF = FF.to(device)

    # ----- Cohen-class reference (target for moments) -----------------------
    pWVD = compute_pWVD(x, N, device)

    # Compute target moments from the Cohen-class reference
    target_mean_t, target_mean_f, target_var_t, target_var_f, target_cov_tf = \
        compute_moments(pWVD, TT, FF)

    print(f"\nTarget moments (from Cohen-class reference pWVD):")
    print(f"  mean_t  = {target_mean_t.item():.6f}")
    print(f"  mean_f  = {target_mean_f.item():.6f}")
    print(f"  var_t   = {target_var_t.item():.6f}")
    print(f"  var_f   = {target_var_f.item():.6f}")
    print(f"  cov_tf  = {target_cov_tf.item():.6f}")
    print()

    # ----- Model ------------------------------------------------------------
    net = TFD_Network(omega0=30.0).to(device)

    # ----- Training Loop (marginals + moments) ------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    num_epochs = 2000
    lambda_moment = 1.0  # weight for moment loss relative to marginal loss

    moment_history = {"loss": [], "marginal": [], "moment": [],
                      "cov_tf": [], "mean_t": [], "mean_f": []}

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred_flat = net(coords)
        P = pred_flat.view(N, N)

        eps = 1e-8
        P = P / (P.sum() + eps)

        pred_time = P.sum(dim=1)
        pred_freq = P.sum(dim=0)

        # Marginal loss (same as baseline)
        marginal_loss = (
            F.mse_loss(pred_time, time_energy) +
            F.mse_loss(pred_freq, freq_energy)
        )

        # Moment loss (MSE per moment term)
        mean_t, mean_f, var_t, var_f, cov_tf = compute_moments(P, TT, FF)

        moment_loss = (
            (mean_t - target_mean_t) ** 2 +
            (mean_f - target_mean_f) ** 2 +
            (var_t - target_var_t) ** 2 +
            (var_f - target_var_f) ** 2 +
            (cov_tf - target_cov_tf) ** 2
        )

        # Combined loss
        total_loss = marginal_loss + lambda_moment * moment_loss

        total_loss.backward()
        optimizer.step()

        # Sanity checks
        assert torch.isfinite(P).all(), "NaN or Inf in P"
        assert torch.isfinite(total_loss), "NaN or Inf in loss"

        # Record history
        if epoch % 10 == 0:
            moment_history["loss"].append(total_loss.item())
            moment_history["marginal"].append(marginal_loss.item())
            moment_history["moment"].append(moment_loss.item())
            moment_history["cov_tf"].append(cov_tf.item())
            moment_history["mean_t"].append(mean_t.item())
            moment_history["mean_f"].append(mean_f.item())

        # Logging
        if epoch % 200 == 0:
            with torch.no_grad():
                rel_err_t = (pred_time - time_energy).norm() / \
                    time_energy.norm()
                rel_err_f = (pred_freq - freq_energy).norm() / \
                    freq_energy.norm()
                mass = P.sum().item()
                print(
                    f"Epoch {epoch:4d} | "
                    f"Total: {total_loss.item():.2e} | "
                    f"Marginal: {marginal_loss.item():.2e} | "
                    f"Moment: {moment_loss.item():.2e} | "
                    f"cov_tf: {cov_tf.item():.4f} | "
                    f"rel_err_t: {rel_err_t.item():.4f} | "
                    f"rel_err_f: {rel_err_f.item():.4f}"
                )

    # ----- Final metrics ----------------------------------------------------
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
        mean_t, mean_f, var_t, var_f, cov_tf = compute_moments(P, TT, FF)
        moment_loss = (
            (mean_t - target_mean_t) ** 2 +
            (mean_f - target_mean_f) ** 2 +
            (var_t - target_var_t) ** 2 +
            (var_f - target_var_f) ** 2 +
            (cov_tf - target_cov_tf) ** 2
        )

    # ----- Visualisation ----------------------------------------------------
    fig = plt.figure(figsize=(20, 10))

    # Panel 1 (top-left) -- Marginal comparison
    ax_t1 = fig.add_subplot(2, 3, 1)
    ax_t1.plot(t_grid.cpu(), time_energy.cpu(), color="steelblue",
               linewidth=1.5, label="True")
    ax_t1.plot(t_grid.cpu(), pred_time.detach().cpu(), "--",
               color="firebrick", linewidth=1.5, label="Predicted")
    ax_t1.set_title(f"Time marginal (rel err = {rel_err_t:.4f})", fontsize=9)
    ax_t1.legend(fontsize=8)
    ax_t1.set_xlabel("t")

    ax_t2 = fig.add_subplot(2, 3, 4)
    ax_t2.plot(f_norm.cpu(), freq_energy.cpu(), color="steelblue",
               linewidth=1.5, label="True")
    ax_t2.plot(f_norm.cpu(), pred_freq.detach().cpu(), "--",
               color="firebrick", linewidth=1.5, label="Predicted")
    ax_t2.set_title(f"Freq marginal (rel err = {rel_err_f:.4f})", fontsize=9)
    ax_t2.legend(fontsize=8)
    ax_t2.set_xlabel("f (normalised)")

    # Panel 2 (top-right) -- Learned P(t, f) with moments
    P_np = P.detach().cpu().numpy()
    vmin2 = np.percentile(P_np, 1)
    vmax2 = np.percentile(P_np, 99)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(P_np.T, origin="lower", aspect="auto",
                     cmap="viridis", vmin=vmin2, vmax=vmax2)
    ax2.set_xlabel("Time index")
    ax2.set_ylabel("Frequency index")
    ax2.set_title("Learned P(t,f) — Marginals + Moments")
    plt.colorbar(im2, ax=ax2, label="Energy density")

    # Panel 3 (top-right) -- Cohen-class reference
    pWVD_np = pWVD.cpu().numpy()
    vmin3 = np.percentile(pWVD_np, 1)
    vmax3 = np.percentile(pWVD_np, 99)

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(pWVD_np.T, origin="lower", aspect="auto",
                     cmap="viridis", vmin=vmin3, vmax=vmax3)
    ax3.set_xlabel("Time index")
    ax3.set_ylabel("Frequency index")
    ax3.set_title("Cohen-Class Reference (smoothed pWVD)")
    plt.colorbar(im3, ax=ax3, label="Energy density")

    # Panel 4 (bottom-middle) -- Moment convergence
    ax4 = fig.add_subplot(2, 3, 5)
    epochs_axis = np.arange(0, num_epochs, 10)
    ax4.semilogy(epochs_axis, moment_history["marginal"],
                 label="Marginal loss", color="steelblue", linewidth=1.2)
    ax4.semilogy(epochs_axis, moment_history["moment"],
                 label="Moment loss", color="firebrick", linewidth=1.2)
    ax4.semilogy(epochs_axis, moment_history["loss"],
                 label="Total loss", color="black", linewidth=1.2,
                 linestyle="--")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss (log scale)")
    ax4.set_title("Loss convergence")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5 (bottom-right) -- Moment values over training
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(epochs_axis, moment_history["cov_tf"],
             label=f"cov_tf (target={target_cov_tf.item():.4f})",
             color="darkorange", linewidth=1.5)
    ax5.axhline(y=target_cov_tf.item(), color="darkorange",
                linestyle=":", alpha=0.7)
    ax5.plot(epochs_axis, moment_history["mean_t"],
             label=f"mean_t (target={target_mean_t.item():.4f})",
             color="steelblue", linewidth=1.2)
    ax5.axhline(y=target_mean_t.item(), color="steelblue",
                linestyle=":", alpha=0.7)
    ax5.plot(epochs_axis, moment_history["mean_f"],
             label=f"mean_f (target={target_mean_f.item():.4f})",
             color="seagreen", linewidth=1.2)
    ax5.axhline(y=target_mean_f.item(), color="seagreen",
                linestyle=":", alpha=0.7)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Moment value")
    ax5.set_title("Moment tracking over training")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("moment_experiment_result.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to moment_experiment_result.png")

    # ----- Result Summary ---------------------------------------------------
    print(f"""
========== MOMENT EXPERIMENT SUMMARY ==========
Marginal loss         : {marginal_loss:.6e}
Moment loss           : {moment_loss:.6e}
Rel error (time)      : {rel_err_t:.6f}
Rel error (freq)      : {rel_err_f:.6f}
Mass of learned P     : {mass:.6f}

Learned moments:
  mean_t  = {mean_t.item():.6f}  (target: {target_mean_t.item():.6f})
  mean_f  = {mean_f.item():.6f}  (target: {target_mean_f.item():.6f})
  var_t   = {var_t.item():.6f}  (target: {target_var_t.item():.6f})
  var_f   = {var_f.item():.6f}  (target: {target_var_f.item():.6f})
  cov_tf  = {cov_tf.item():.6f}  (target: {target_cov_tf.item():.6f})

Diagnosis: Moment constraints add time-frequency correlation
information that marginals alone cannot provide. The covariance
term cov(t,f) encodes the linear relationship between time and
frequency that characterises the chirp. Compare the learned P(t,f)
with the marginal-only baseline to assess whether moments help
recover the diagonal ridge structure.
=================================================""")


if __name__ == "__main__":
    main()
