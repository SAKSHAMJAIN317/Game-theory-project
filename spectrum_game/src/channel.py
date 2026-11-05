"""Wireless channel, path loss, and network topology generation."""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import SystemConfig


def calculate_pl0(f_c: float = 3.5e9, c: float = 3.0e8) -> float:
    """Calculate free-space reference path loss at d_0 = 1m: PL0 = 20*log10(4*pi*1*f_c/c)."""
    return 20.0 * math.log10(4.0 * math.pi * 1.0 * f_c / c)


@dataclass
class NetworkTopology:
    """Represents a generated network topology with channel gains and interference matrices."""
    config: SystemConfig
    seed: int
    tx_pos: np.ndarray  # Shape: (N, 2)
    rx_pos: np.ndarray  # Shape: (N, 2)
    dist: np.ndarray    # Shape: (N, N), dist[j, i] = distance from Tx j to Rx i
    PL_dB: np.ndarray   # Shape: (N, N), path loss in dB
    G: np.ndarray       # Shape: (N, N), linear power gain g[j, i] from Tx j to Rx i
    W: np.ndarray       # Shape: (N, N), symmetrized interference matrix W[i, j]

    @property
    def N(self) -> int:
        return self.config.N

    @property
    def K(self) -> int:
        return self.config.K

    def compute_interference(self, a: np.ndarray) -> np.ndarray:
        """
        Compute total co-channel interference power at each receiver i in mW:
        I_i = sum_{j != i, a_j == a_i} P * g[j, i]
        """
        a = np.asarray(a)
        P = self.config.P_mW
        # Mask where a_j == a_i, with diagonal false (j != i)
        same_channel = (a[:, np.newaxis] == a[np.newaxis, :])  # shape (N, N), [j, i]
        np.fill_diagonal(same_channel, False)
        # G[j, i] is gain Tx j -> Rx i. Co-channel Tx j delivers P * G[j, i] to Rx i
        interf = np.sum(P * self.G * same_channel, axis=0)  # sum over j
        return interf

    def compute_sinr(self, a: np.ndarray) -> np.ndarray:
        """
        Compute SINR for each link i:
        SINR_i = P * g[i, i] / (sigma2 + sum_{j != i, a_j == a_i} P * g[j, i])
        """
        a = np.asarray(a)
        P = self.config.P_mW
        sigma2 = self.config.sigma2_mW
        S = P * np.diag(self.G)
        I = self.compute_interference(a)
        return S / (sigma2 + I)

    def compute_rates(self, a: np.ndarray) -> np.ndarray:
        """
        Compute Shannon achievable rate for each link i in bps:
        R_i = B * log2(1 + SINR_i)
        """
        sinr = self.compute_sinr(a)
        return self.config.B * np.log2(1.0 + sinr)

    def compute_rates_mbps(self, a: np.ndarray) -> np.ndarray:
        """Compute rates in Mbps."""
        return self.compute_rates(a) / 1.0e6

    def compute_potential(self, a: np.ndarray) -> float:
        """
        Compute exact potential function for G1:
        Phi(a) = - sum_{i < j} W[i, j] * [a_i == a_j]
        """
        a = np.asarray(a)
        same_channel = (a[:, np.newaxis] == a[np.newaxis, :])
        # Upper triangular mask (i < j)
        i_less_j = np.triu(np.ones((self.N, self.N), dtype=bool), k=1)
        active_pairs = same_channel & i_less_j
        return -float(np.sum(self.W[active_pairs]))

    def compute_interference_cost(self, a: np.ndarray) -> float:
        """
        Interference cost C(a) = - Phi(a) = sum_{i < j} W[i, j] * [a_i == a_j] >= 0.
        """
        return -self.compute_potential(a)

    def compute_utility_g1(self, i: int, channel: int, a: np.ndarray) -> float:
        """
        Compute utility of player i if taking 'channel', given others' actions a_{-i}:
        u_i(channel, a_{-i}) = - sum_{j != i, a_j == channel} W[i, j]
        """
        # Vectorized over other players j != i
        other_mask = (a == channel)
        other_mask[i] = False
        return -float(np.sum(self.W[i, other_mask]))

    def compute_utility_g2(self, i: int, channel: int, a: np.ndarray) -> float:
        """
        Compute individual rate utility of player i if taking 'channel' in G2:
        u_i(channel, a_{-i}) = R_i(channel, a_{-i})
        """
        a_test = a.copy()
        a_test[i] = channel
        P = self.config.P_mW
        sigma2 = self.config.sigma2_mW
        # Interference at receiver i from transmitters j != i on same channel
        mask = (a_test == channel)
        mask[i] = False
        interf = np.sum(P * self.G[mask, i])
        sinr_i = (P * self.G[i, i]) / (sigma2 + interf)
        return float(self.config.B * math.log2(1.0 + sinr_i))


def generate_topology(
    config: Optional[SystemConfig] = None,
    seed: int = 42,
    n_pathloss: Optional[float] = None,
    sigma_shadowing: Optional[float] = None,
) -> NetworkTopology:
    """
    Generate a random 5G/6G network topology with N Tx-Rx links in an L x L area.
    
    Model parameters:
    - Tx locations uniform in [0, L] x [0, L].
    - Rx locations at distance U[rx_dist_min, rx_dist_max] and angle U[0, 2*pi).
    - PL_dB(d) = PL0 + 10*n*log10(max(d, 1)) + X, X ~ N(0, sigma_sh^2) per (Tx j, Rx i).
    - g[j, i] = 10^(-PL/10).
    - W[i, j] = P * (g[i, j] + g[j, i]) / sigma2, diagonal 0.
    """
    if config is None:
        config = SystemConfig()

    n = n_pathloss if n_pathloss is not None else config.n_pathloss
    sigma_sh = sigma_shadowing if sigma_shadowing is not None else config.sigma_shadowing

    rng = np.random.default_rng(seed)
    N = config.N
    L = config.L

    # 1. Transmitter coordinates: uniform in [0, L] x [0, L]
    tx_pos = rng.uniform(0.0, L, size=(N, 2))

    # 2. Receiver coordinates: distance U[d_min, d_max] and angle U[0, 2*pi)
    d_rx = rng.uniform(config.rx_dist_min, config.rx_dist_max, size=N)
    theta_rx = rng.uniform(0.0, 2.0 * math.pi, size=N)
    rx_pos = np.zeros((N, 2))
    rx_pos[:, 0] = tx_pos[:, 0] + d_rx * np.cos(theta_rx)
    rx_pos[:, 1] = tx_pos[:, 1] + d_rx * np.sin(theta_rx)

    # 3. Distance matrix: dist[j, i] = distance from Tx j to Rx i
    diff = tx_pos[:, np.newaxis, :] - rx_pos[np.newaxis, :, :]  # (N, N, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)

    # 4. Shadowing matrix: X[j, i] ~ N(0, sigma_sh^2) independent per (Tx j, Rx i)
    if sigma_sh > 0:
        X = rng.normal(0.0, sigma_sh, size=(N, N))
    else:
        X = np.zeros((N, N))

    # 5. Path loss: PL_dB(d) = PL0 + 10*n*log10(max(d, 1)) + X
    pl0 = config.PL0_dB
    PL_dB = pl0 + 10.0 * n * np.log10(np.maximum(dist, 1.0)) + X

    # 6. Channel gain: g[j, i] = 10^(-PL / 10)
    G = 10.0 ** (-PL_dB / 10.0)

    # 7. Symmetrized interference matrix W:
    # W[i, j] = P * (g[i, j] + g[j, i]) / sigma2, diagonal 0
    P = config.P_mW
    sigma2 = config.sigma2_mW
    W = (P / sigma2) * (G + G.T)
    np.fill_diagonal(W, 0.0)

    return NetworkTopology(
        config=config,
        seed=seed,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        dist=dist,
        PL_dB=PL_dB,
        G=G,
        W=W,
    )
