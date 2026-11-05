"""Visualization module generating publication-quality figures at 200 DPI."""

import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set modern style defaults
mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Curated qualitative palette for channels and methods
PALETTE = [
    "#2b5c8f",  # Deep Blue
    "#d95f02",  # Vibrant Orange
    "#2ca02c",  # Forest Green
    "#9467bd",  # Royal Purple
    "#e377c2",  # Rose Pink
    "#8c564b",  # Chestnut
    "#17becf",  # Bright Cyan
    "#bcbd22",  # Olive
]

METHOD_COLORS = {
    "Proposed NE (G1)": "#2b5c8f",
    "G2 Rate Game": "#9467bd",
    "One-Pass Greedy": "#2ca02c",
    "Round-Robin": "#d95f02",
    "Random": "#7f7f7f",
    "Random (Averaged 50 draws)": "#7f7f7f",
    "Exhaustive (Best Phi)": "#17becf",
    "Exhaustive (Best Sum-Rate)": "#e377c2",
}


def _ensure_dir(filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


def plot_topology_before_after(
    topology,
    initial_a: np.ndarray,
    final_a: np.ndarray,
    save_path: str = "results/figures/fig1_topology_before_after.png",
) -> None:
    """Plot network topology with links colored by initial vs final channel allocation."""
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    N = topology.N
    K = topology.K
    cmap = plt.cm.get_cmap("tab10", K)

    # Subplot 1: Initial Allocation
    for i in range(N):
        ch = initial_a[i]
        c = cmap(ch)
        ax1.plot(
            [topology.tx_pos[i, 0], topology.rx_pos[i, 0]],
            [topology.tx_pos[i, 1], topology.rx_pos[i, 1]],
            color=c,
            alpha=0.6,
            linewidth=1.8,
            linestyle="--",
        )
        ax1.scatter(topology.tx_pos[i, 0], topology.tx_pos[i, 1], color=c, marker="^", s=70, edgecolors="k", zorder=3)
        ax1.scatter(topology.rx_pos[i, 0], topology.rx_pos[i, 1], color=c, marker="o", s=50, edgecolors="k", zorder=3)
        ax1.annotate(f"{i}", (topology.tx_pos[i, 0] + 3, topology.tx_pos[i, 1] + 3), fontsize=8)

    ax1.set_title(f"Initial State (Random Allocation, N={N}, K={K})", fontweight="bold")
    ax1.set_xlabel("X Coordinate (m)")
    ax1.set_ylabel("Y Coordinate (m)")
    ax1.set_xlim(-10, topology.config.L + 30)
    ax1.set_ylim(-10, topology.config.L + 30)

    # Subplot 2: Final NE Allocation
    for i in range(N):
        ch = final_a[i]
        c = cmap(ch)
        ax2.plot(
            [topology.tx_pos[i, 0], topology.rx_pos[i, 0]],
            [topology.tx_pos[i, 1], topology.rx_pos[i, 1]],
            color=c,
            alpha=0.7,
            linewidth=2.0,
        )
        ax2.scatter(topology.tx_pos[i, 0], topology.tx_pos[i, 1], color=c, marker="^", s=70, edgecolors="k", zorder=3)
        ax2.scatter(topology.rx_pos[i, 0], topology.rx_pos[i, 1], color=c, marker="o", s=50, edgecolors="k", zorder=3)
        ax2.annotate(f"{i}", (topology.tx_pos[i, 0] + 3, topology.tx_pos[i, 1] + 3), fontsize=8)

    # Custom legend for channels
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(k), markersize=8, label=f"Channel {k}")
        for k in range(K)
    ]
    handles.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label="Tx"))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, label="Rx"))
    ax2.legend(handles=handles, loc="upper right", framealpha=0.9)

    ax2.set_title(f"Converged State (G1 Nash Equilibrium, N={N}, K={K})", fontweight="bold")
    ax2.set_xlabel("X Coordinate (m)")
    ax2.set_ylabel("Y Coordinate (m)")
    ax2.set_xlim(-10, topology.config.L + 30)
    ax2.set_ylim(-10, topology.config.L + 30)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_convergence_phi(
    history,
    save_path: str = "results/figures/fig2_phi_vs_iteration.png",
) -> None:
    """Plot Potential function Phi vs sequential switch iteration (verifying monotonicity)."""
    _ensure_dir(save_path)
    steps = [h.step for h in history]
    phis = [h.phi for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, phis, marker="o", color="#2b5c8f", linewidth=2.0, markersize=5, label=r"Potential $\Phi(a)$")
    ax.set_title(r"Potential Function $\Phi(a)$ vs Switch Step (Monotonic Non-Decreasing)", fontweight="bold")
    ax.set_xlabel("Sequential Switch Step")
    ax.set_ylabel(r"Exact Potential $\Phi(a)$ [$-1 \times$ Interference Cost]")
    ax.legend(loc="lower right")

    # Annotate initial and converged points
    ax.annotate(
        f"Init: {phis[0]:.2e}",
        xy=(steps[0], phis[0]),
        xytext=(steps[0] + 0.5, phis[0] * 1.1),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=9,
    )
    ax.annotate(
        f"NE: {phis[-1]:.2e}",
        xy=(steps[-1], phis[-1]),
        xytext=(max(0, steps[-1] - 3), phis[-1] * 0.9),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_convergence_sum_rate(
    history,
    save_path: str = "results/figures/fig3_sum_rate_vs_iteration.png",
) -> None:
    """Plot network sum-rate evolution vs sequential switch iteration."""
    _ensure_dir(save_path)
    steps = [h.step for h in history]
    sum_rates = [h.sum_rate_mbps for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, sum_rates, marker="s", color="#2ca02c", linewidth=2.0, markersize=5, label="Network Sum-Rate")
    ax.set_title("Network Sum-Rate Evolution during Best-Response Dynamics", fontweight="bold")
    ax.set_xlabel("Sequential Switch Step")
    ax.set_ylabel("Sum-Rate (Mbps)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_method_bar_chart(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig4_method_bar_chart.png",
) -> None:
    """Compare methods across key metrics (Sum Rate, Jain Index, Outage Fraction) with error bars."""
    _ensure_dir(save_path)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    methods = df["method"].unique()
    colors = [METHOD_COLORS.get(m, "#4a7bb0") for m in methods]

    # Group by method to get mean and 95% CI
    summary = df.groupby("method").agg(
        sum_rate_mean=("sum_rate_mbps", "mean"),
        sum_rate_std=("sum_rate_mbps", "std"),
        sum_rate_count=("sum_rate_mbps", "count"),
        jain_mean=("jain_index", "mean"),
        jain_std=("jain_index", "std"),
        outage_mean=("outage_fraction", "mean"),
        outage_std=("outage_fraction", "std"),
    ).loc[methods]

    # 95% CI margins: 1.96 * std / sqrt(n)
    sr_ci = 1.96 * summary["sum_rate_std"] / np.sqrt(summary["sum_rate_count"])
    jain_ci = 1.96 * summary["jain_std"] / np.sqrt(summary["sum_rate_count"])
    outage_ci = 1.96 * summary["outage_std"] / np.sqrt(summary["sum_rate_count"])

    x = np.arange(len(methods))

    # Ax1: Sum Rate
    ax1.bar(x, summary["sum_rate_mean"], yerr=sr_ci, capsize=5, color=colors, alpha=0.85, edgecolor="k")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right")
    ax1.set_ylabel("Sum-Rate (Mbps)")
    ax1.set_title("Mean Sum-Rate (± 95% CI)", fontweight="bold")

    # Ax2: Jain Index
    ax2.bar(x, summary["jain_mean"], yerr=jain_ci, capsize=5, color=colors, alpha=0.85, edgecolor="k")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right")
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Jain's Fairness Index (± 95% CI)", fontweight="bold")

    # Ax3: Outage Fraction
    ax3.bar(x, summary["outage_mean"] * 100.0, yerr=outage_ci * 100.0, capsize=5, color=colors, alpha=0.85, edgecolor="k")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=25, ha="right")
    ax3.set_ylabel("Outage Fraction (%)")
    ax3.set_title("Outage Fraction [SINR < 0 dB]", fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_rate_cdf(
    rates_dict: Dict[str, np.ndarray],
    save_path: str = "results/figures/fig5_rate_cdf.png",
) -> None:
    """Plot empirical Cumulative Distribution Function (CDF) of link rates across methods."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, rates in rates_dict.items():
        sorted_rates = np.sort(rates)
        cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
        color = METHOD_COLORS.get(method, None)
        ax.plot(sorted_rates, cdf, label=method, linewidth=2.0, color=color)

    ax.set_title("Empirical CDF of Link Rates (Mbps)", fontweight="bold")
    ax.set_xlabel("Link Achievable Rate (Mbps)")
    ax.set_ylabel("Empirical Probability P(Rate ≤ R)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_scaling_n(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig6_sum_rate_jain_vs_n.png",
) -> None:
    """Plot Sum-Rate and Jain Index scaling with network density N."""
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in df["method"].unique():
        sub = df[df["method"] == method].groupby("N").agg(
            sr_mean=("sum_rate_mbps", "mean"),
            sr_std=("sum_rate_mbps", "std"),
            sr_count=("sum_rate_mbps", "count"),
            jain_mean=("jain_index", "mean"),
            jain_std=("jain_index", "std"),
        ).reset_index()

        sr_ci = 1.96 * sub["sr_std"] / np.sqrt(sub["sr_count"])
        jain_ci = 1.96 * sub["jain_std"] / np.sqrt(sub["sr_count"])
        c = METHOD_COLORS.get(method, None)

        ax1.plot(sub["N"], sub["sr_mean"], marker="o", label=method, color=c, linewidth=2.0)
        ax1.fill_between(sub["N"], sub["sr_mean"] - sr_ci, sub["sr_mean"] + sr_ci, color=c, alpha=0.15)

        ax2.plot(sub["N"], sub["jain_mean"], marker="s", label=method, color=c, linewidth=2.0)
        ax2.fill_between(sub["N"], sub["jain_mean"] - jain_ci, sub["jain_mean"] + jain_ci, color=c, alpha=0.15)

    ax1.set_title("Network Sum-Rate vs Link Density N (K=4)", fontweight="bold")
    ax1.set_xlabel("Number of Links N")
    ax1.set_ylabel("Sum-Rate (Mbps)")
    ax1.legend(loc="upper left")

    ax2.set_title("Jain's Fairness Index vs Link Density N (K=4)", fontweight="bold")
    ax2.set_xlabel("Number of Links N")
    ax2.set_ylabel("Jain's Index")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_scaling_k(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig7_sum_rate_vs_k.png",
) -> None:
    """Plot Sum-Rate scaling with number of orthogonal channels K."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in df["method"].unique():
        sub = df[df["method"] == method].groupby("K").agg(
            sr_mean=("sum_rate_mbps", "mean"),
            sr_std=("sum_rate_mbps", "std"),
            sr_count=("sum_rate_mbps", "count"),
        ).reset_index()

        sr_ci = 1.96 * sub["sr_std"] / np.sqrt(sub["sr_count"])
        c = METHOD_COLORS.get(method, None)
        ax.plot(sub["K"], sub["sr_mean"], marker="o", label=method, color=c, linewidth=2.0)
        ax.fill_between(sub["K"], sub["sr_mean"] - sr_ci, sub["sr_mean"] + sr_ci, color=c, alpha=0.15)

    ax.set_title("Network Sum-Rate vs Number of Channels K (N=20)", fontweight="bold")
    ax.set_xlabel("Number of Channels K")
    ax.set_ylabel("Sum-Rate (Mbps)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_rounds_boxplot(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig8_rounds_vs_n_boxplot.png",
) -> None:
    """Boxplot of Best-Response convergence rounds vs network size N."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    g1_df = df[df["method"] == "Proposed NE (G1)"]
    n_values = sorted(g1_df["N"].unique())
    data = [g1_df[g1_df["N"] == n]["rounds"].values for n in n_values]

    bp = ax.boxplot(
        data,
        tick_labels=n_values,
        patch_artist=True,
        medianprops=dict(color="k", linewidth=1.5),
        boxprops=dict(facecolor="#9ecae1", color="#2b5c8f", alpha=0.8),
    )

    ax.set_title("Best-Response Convergence Rounds vs Network Density N", fontweight="bold")
    ax.set_xlabel("Number of Links N")
    ax.set_ylabel("Convergence Rounds")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_channel_load_histogram(
    allocations_dict: Dict[str, List[np.ndarray]],
    K: int,
    save_path: str = "results/figures/fig9_channel_load_histogram.png",
) -> None:
    """Distribution of channel loads across orthogonal channels."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5))

    channels = np.arange(K)
    width = 0.8 / len(allocations_dict)

    for idx, (method, allocs) in enumerate(allocations_dict.items()):
        # Compute average links per channel across topologies
        load_counts = np.zeros(K)
        for a in allocs:
            counts = np.bincount(a, minlength=K)
            load_counts += counts
        mean_loads = load_counts / len(allocs)

        x = channels - 0.4 + idx * width + width / 2
        c = METHOD_COLORS.get(method, None)
        ax.bar(x, mean_loads, width=width, label=method, color=c, alpha=0.85, edgecolor="k")

    ax.set_title(f"Channel Load Distribution Across {K} Orthogonal Channels", fontweight="bold")
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Average Assigned Links")
    ax.set_xticks(channels)
    ax.set_xticklabels([f"Ch {k}" for k in channels])
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_optimality_gaps(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig10_optimality_gap_histogram.png",
) -> None:
    """Distribution of optimality gaps for N=8, K=3 exhaustive comparison."""
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Filter Proposed NE
    g1_df = df[df["method"] == "Proposed NE (G1)"]
    pot_diffs = g1_df["potential_diff"].dropna().values
    sr_gaps = g1_df["sum_rate_gap_rel"].dropna().values * 100.0  # percentage

    ax1.hist(pot_diffs, bins=15, color="#2b5c8f", edgecolor="k", alpha=0.8)
    ax1.set_title(r"Potential Gap: $\Phi^* - \Phi(a_{NE})$ (Non-Negative)", fontweight="bold")
    ax1.set_xlabel(r"Potential Difference $\Delta \Phi$")
    ax1.set_ylabel("Frequency (Topologies)")

    ax2.hist(sr_gaps, bins=15, color="#2ca02c", edgecolor="k", alpha=0.8)
    ax2.set_title(r"Sum-Rate Optimality Gap: $\frac{R^*_{sum} - R_{NE}}{R^*_{sum}}$ (%)", fontweight="bold")
    ax2.set_xlabel("Sum-Rate Optimality Gap (%)")
    ax2.set_ylabel("Frequency (Topologies)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_sensitivity(
    df: pd.DataFrame,
    save_path: str = "results/figures/fig11_sensitivity_plot.png",
) -> None:
    """Sensitivity analysis across path loss exponent n and shadowing sigma_sh."""
    _ensure_dir(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    g1_df = df[df["method"] == "Proposed NE (G1)"]

    # 1. Path loss exponent n
    n_group = g1_df.groupby("n_pathloss").agg(
        sr_mean=("sum_rate_mbps", "mean"),
        sr_std=("sum_rate_mbps", "std"),
        sr_count=("sum_rate_mbps", "count"),
    ).reset_index()
    n_ci = 1.96 * n_group["sr_std"] / np.sqrt(n_group["sr_count"])
    ax1.errorbar(
        n_group["n_pathloss"],
        n_group["sr_mean"],
        yerr=n_ci,
        marker="o",
        color="#2b5c8f",
        capsize=5,
        linewidth=2.0,
    )
    ax1.set_title("Sum-Rate vs Path Loss Exponent n", fontweight="bold")
    ax1.set_xlabel("Path Loss Exponent n")
    ax1.set_ylabel("Sum-Rate (Mbps)")

    # 2. Shadowing std dev sigma_sh
    sh_group = g1_df.groupby("sigma_shadowing").agg(
        sr_mean=("sum_rate_mbps", "mean"),
        sr_std=("sum_rate_mbps", "std"),
        sr_count=("sum_rate_mbps", "count"),
    ).reset_index()
    sh_ci = 1.96 * sh_group["sr_std"] / np.sqrt(sh_group["sr_count"])
    ax2.errorbar(
        sh_group["sigma_shadowing"],
        sh_group["sr_mean"],
        yerr=sh_ci,
        marker="s",
        color="#d95f02",
        capsize=5,
        linewidth=2.0,
    )
    ax2.set_title(r"Sum-Rate vs Shadowing Deviation $\sigma_{sh}$ (dB)", fontweight="bold")
    ax2.set_xlabel(r"Shadowing Deviation $\sigma_{sh}$ (dB)")
    ax2.set_ylabel("Sum-Rate (Mbps)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
