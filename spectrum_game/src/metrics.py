"""Metrics calculation, fairness indices, and statistical significance testing."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats

from .channel import NetworkTopology


@dataclass
class AllocationMetrics:
    method: str
    seed: int
    N: int
    K: int
    sum_rate_mbps: float
    mean_rate_mbps: float
    p5_rate_mbps: float
    total_inr: float
    mean_inr_dB: float
    co_channel_pairs: int
    jain_index: float
    outage_fraction: float
    load_balance_std: float
    phi: float
    interference_cost: float
    rounds: int = 1
    switches: int = 0
    converged: bool = True
    is_nash: bool = False
    runtime_sec: float = 0.0
    # Optimality gaps (if optimum known)
    potential_diff: Optional[float] = None
    cost_gap_rel: Optional[float] = None
    sum_rate_gap_rel: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_jain_index(rates: np.ndarray) -> float:
    """
    Jain's Fairness Index:
    J = (sum(R_i))^2 / (N * sum(R_i^2))
    """
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    if n == 0:
        return 1.0
    sum_r = float(np.sum(rates))
    sum_sq_r = float(np.sum(rates ** 2))
    if sum_sq_r <= 1.0e-12:
        return 1.0
    return float((sum_r ** 2) / (n * sum_sq_r))


def compute_metrics(
    topology: NetworkTopology,
    allocation: np.ndarray,
    method_name: str,
    rounds: int = 1,
    switches: int = 0,
    converged: bool = True,
    is_nash: bool = False,
    runtime_sec: float = 0.0,
    opt_phi: Optional[float] = None,
    opt_sum_rate: Optional[float] = None,
) -> AllocationMetrics:
    """
    Compute full suite of metrics for a given allocation vector.
    """
    a = np.asarray(allocation)
    N = topology.N
    K = topology.K

    # 1. Rates
    rates_mbps = topology.compute_rates_mbps(a)
    sum_rate = float(np.sum(rates_mbps))
    mean_rate = float(np.mean(rates_mbps))
    p5_rate = float(np.percentile(rates_mbps, 5))

    # 2. SINR & Outage (outage defined as SINR < 0 dB, i.e. linear SINR < 1.0)
    sinr_linear = topology.compute_sinr(a)
    outage_frac = float(np.mean(sinr_linear < 1.0))

    # 3. Interference & INR
    I = topology.compute_interference(a)
    sigma2 = topology.config.sigma2_mW
    inr_linear = I / sigma2
    total_inr = float(np.sum(inr_linear))
    mean_inr = float(np.mean(inr_linear))
    mean_inr_dB = float(10.0 * np.log10(max(mean_inr, 1.0e-12)))

    # 4. Co-channel pair count: sum_{i < j} [a_i == a_j]
    same_channel = (a[:, np.newaxis] == a[np.newaxis, :])
    i_less_j = np.triu(np.ones((N, N), dtype=bool), k=1)
    co_channel_pairs = int(np.sum(same_channel & i_less_j))

    # 5. Jain Index
    jain = calculate_jain_index(rates_mbps)

    # 6. Load-balance std across K channels
    counts = np.bincount(a, minlength=K)
    load_std = float(np.std(counts))

    # 7. Potential & Interference Cost
    phi = topology.compute_potential(a)
    cost = -phi

    # 8. Optimality gaps
    pot_diff = None
    cost_gap = None
    sr_gap = None

    if opt_phi is not None:
        # Note: Phi is negative interference cost (Phi = -C).
        # opt_phi is max Phi (minimal cost C* = -opt_phi).
        # Potential difference: opt_phi - phi >= 0.
        pot_diff = float(opt_phi - phi)
        opt_cost = -opt_phi
        denom = max(opt_cost, 1.0e-9)
        cost_gap = float((cost - opt_cost) / denom)

    if opt_sum_rate is not None and opt_sum_rate > 1.0e-9:
        sr_gap = float((opt_sum_rate - sum_rate) / opt_sum_rate)

    return AllocationMetrics(
        method=method_name,
        seed=topology.seed,
        N=N,
        K=K,
        sum_rate_mbps=sum_rate,
        mean_rate_mbps=mean_rate,
        p5_rate_mbps=p5_rate,
        total_inr=total_inr,
        mean_inr_dB=mean_inr_dB,
        co_channel_pairs=co_channel_pairs,
        jain_index=jain,
        outage_fraction=outage_frac,
        load_balance_std=load_std,
        phi=phi,
        interference_cost=cost,
        rounds=rounds,
        switches=switches,
        converged=converged,
        is_nash=is_nash,
        runtime_sec=runtime_sec,
        potential_diff=pot_diff,
        cost_gap_rel=cost_gap,
        sum_rate_gap_rel=sr_gap,
    )


def compute_summary_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, and 95% Confidence Interval margin."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "p5": 0.0, "p95": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    # 95% CI margin
    if n > 1:
        sem = std / np.sqrt(n)
        ci95 = float(stats.t.ppf(0.975, df=n - 1) * sem)
    else:
        ci95 = 0.0

    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def perform_wilcoxon_paired_test(
    proposed: List[float],
    baseline: List[float],
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank paired test between proposed and baseline.
    Returns (statistic, p_value).
    """
    diff = np.asarray(proposed) - np.asarray(baseline)
    # Check if all differences are zero
    if np.all(np.abs(diff) < 1.0e-12):
        return 0.0, 1.0

    # Non-zero differences only
    try:
        res = stats.wilcoxon(proposed, baseline)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0
