from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# -------------------------
# Fig.11-like: Real vs Pseudo waveform
# -------------------------
def plot_signal_comparison(
    real: torch.Tensor,
    pseudo: torch.Tensor,
    epoch: int,
    save_dir: Path,
    sample_idx: int = 0,
    fs: Optional[float] = None,
    max_seconds: Optional[float] = 0.1,
    ylabel: str = r"Amplitude (m/s$^2$)",
    legend: Tuple[str, str] = ("Real", "Pseudo"),
    filename_prefix: str = "real_vs_pseudo",
) -> None:
    """
    Plot a single sample of real vs pseudo/reconstructed signal.

    Args:
        real, pseudo: (B,T) or (B,1,T) or (B,T,*) flattened to 1D
        fs: sampling frequency. If provided, x-axis is Time (s); otherwise index.
        max_seconds: if fs is provided, crop to at most this many seconds.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    r = real[sample_idx].detach().cpu().flatten().numpy()
    p = pseudo[sample_idx].detach().cpu().flatten().numpy()
    T = min(len(r), len(p))
    r = r[:T]
    p = p[:T]

    if fs is not None and fs > 0:
        t = np.arange(T, dtype=np.float32) / float(fs)
        if max_seconds is not None and max_seconds > 0:
            n = int(max_seconds * float(fs))
            n = max(1, min(n, T))
            t = t[:n]
            r = r[:n]
            p = p[:n]
        x = t
        xlabel = "Time (s)"
    else:
        x = np.arange(T, dtype=np.int32)
        xlabel = "Time index"

    plt.figure(figsize=(7.2, 3.0))
    plt.plot(x, r, label=legend[0], linewidth=1.0)
    plt.plot(x, p, label=legend[1], linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{filename_prefix}_epoch_{epoch:03d}.png", dpi=200)
    plt.close()


# Backward-compatible alias (old name used by train.py)
def plot_reconstruction_comparison(
    acc_real: torch.Tensor,
    acc_hat: torch.Tensor,
    epoch: int,
    save_dir: Path,
    sample_idx: int = 0,
    fs: Optional[float] = None,
    max_seconds: Optional[float] = 0.1,
) -> None:
    """
    Backward-compatible wrapper. If you pass fs, it will match the Fig.11-like layout.
    """
    plot_signal_comparison(
        real=acc_real,
        pseudo=acc_hat,
        epoch=epoch,
        save_dir=save_dir,
        sample_idx=sample_idx,
        fs=fs,
        max_seconds=max_seconds,
        legend=("Real", "Pseudo"),
        filename_prefix="recon",
    )


# -------------------------
# Fig.9-like: Digital twin space (scatter + marginal histograms)
# -------------------------
def plot_digital_twin_space(
    z_by_label: Mapping[int, np.ndarray],
    save_path: Path,
    *,
    stiffness_index: int = 1,
    damping_index: int = 0,
    title: str = "CWRU digital twin space",
    xlabel: str = "Dimensionless Stiffness",
    ylabel: str = "Dimensionless Damping",
    bins: int = 12,
    point_size: float = 28.0,
    exclude_labels: Sequence[int] = (0,),
) -> None:
    """
    Create a joint scatter plot with marginal histograms.

    Expected z_by_label[label] shape: (N,2) where columns are [c_tilde, k_tilde]
    by default (from train.infer_ck_params). This function maps:
        x = k_tilde (stiffness_index=1)
        y = c_tilde (damping_index=0)

    Args:
        exclude_labels: labels to omit (default excludes normal label 0).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [l for l in sorted(z_by_label.keys()) if l not in set(exclude_labels)]

    # Collect for global histogram ranges
    xs_all, ys_all = [], []
    for l in labels:
        z = np.asarray(z_by_label[l])
        if z.size == 0:
            continue
        xs_all.append(z[:, stiffness_index])
        ys_all.append(z[:, damping_index])
    if not xs_all or not ys_all:
        # nothing to plot
        return

    x_all = np.concatenate(xs_all, axis=0)
    y_all = np.concatenate(ys_all, axis=0)

    # Layout: top hist + right hist + center scatter
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        wspace=0.12,
        hspace=0.12,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    # Colors: stable and distinct
    cmap = plt.get_cmap("tab10")

    handles = []
    for i, l in enumerate(labels):
        z = np.asarray(z_by_label[l])
        if z.size == 0:
            continue
        x = z[:, stiffness_index]
        y = z[:, damping_index]
        color = cmap(i % 10)

        sc = ax_scatter.scatter(x, y, s=point_size, alpha=0.9, color=color, label=str(l))
        handles.append(sc)

    # Marginals (use the pooled data for axis limits consistency)
    ax_histx.hist(x_all, bins=bins)
    ax_histy.hist(y_all, bins=bins, orientation="horizontal")

    # Labels & cosmetics
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_histx.set_ylabel("Frequency")
    ax_histy.set_xlabel("Frequency")

    ax_histx.set_title(title)

    # Hide redundant tick labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Share limits for cleaner look
    def _with_pad(lo: float, hi: float) -> Tuple[float, float]:
        if not np.isfinite(lo) or not np.isfinite(hi):
            return -1.0, 1.0
        if lo == hi:
            pad = max(1e-4, abs(lo) * 0.05)
            return lo - pad, hi + pad
        span = hi - lo
        pad = max(1e-3, span * 0.1)
        return lo - pad, hi + pad

    x_lim = _with_pad(float(np.min(x_all)), float(np.max(x_all)))
    y_lim = _with_pad(float(np.min(y_all)), float(np.max(y_all)))

    ax_scatter.set_xlim(x_lim)
    ax_scatter.set_ylim(y_lim)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    # Legend below the scatter (similar to the paper)
    if handles:
        ax_scatter.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(len(labels), 9),
            frameon=True,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Loss curves (existing)
# -------------------------
def plot_loss_curves(history: Dict[str, Sequence[float]], save_dir: Path) -> None:
    """Plot loss history over epochs."""
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["Ltot"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["Lddl"], label="L_DDL")
    plt.plot(epochs, history["Ldt"], label="L_DT")
    plt.plot(epochs, history["Lrec"], label="L_Rec")
    plt.plot(epochs, history["Ltot"], label="L_total", color="black", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close()
