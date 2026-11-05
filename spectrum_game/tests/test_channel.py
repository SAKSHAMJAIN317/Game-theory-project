"""Unit tests for wireless channel model, path loss, and topology generation."""

import math
import numpy as np
import pytest

from src.config import SystemConfig
from src.channel import calculate_pl0, generate_topology


def test_pl0_formula():
    """Verify PL0 calculation against analytical free-space reference formula at 1m."""
    f_c = 3.5e9
    c = 3.0e8
    expected = 20.0 * math.log10(4.0 * math.pi * 1.0 * f_c / c)
    calculated = calculate_pl0(f_c, c)
    assert math.isclose(calculated, expected, rel_tol=1e-12)
    # Expected is approx 43.328 dB
    assert 43.0 < calculated < 44.0

    # Also test via SystemConfig property
    cfg = SystemConfig(f_c=f_c, c=c)
    assert math.isclose(cfg.PL0_dB, expected, rel_tol=1e-12)


def test_single_user_sinr_equals_snr():
    """Verify that when N=1 (single link), SINR equals SNR exactly (interference is 0)."""
    cfg = SystemConfig(N=1, K=2)
    topo = generate_topology(config=cfg, seed=123)

    allocation = np.array([0])
    sinr = topo.compute_sinr(allocation)
    interf = topo.compute_interference(allocation)

    assert interf[0] == 0.0

    # Expected SNR = P * g[0, 0] / sigma2
    expected_snr = (cfg.P_mW * topo.G[0, 0]) / cfg.sigma2_mW
    assert math.isclose(float(sinr[0]), expected_snr, rel_tol=1e-9)


def test_noise_power_calculation():
    """Verify noise power calculation: sigma2_dBm = -174 + 10*log10(B) + NF."""
    cfg = SystemConfig(B=20.0e6, NF_dB=7.0)
    expected_dBm = -174.0 + 10.0 * math.log10(20.0e6) + 7.0
    assert math.isclose(cfg.sigma2_dBm, expected_dBm, rel_tol=1e-9)
    assert math.isclose(cfg.sigma2_mW, 10.0 ** (expected_dBm / 10.0), rel_tol=1e-9)


def test_interference_matrix_symmetry():
    """Verify that symmetrized interference matrix W is symmetric with zero diagonal."""
    cfg = SystemConfig(N=10, K=3)
    topo = generate_topology(config=cfg, seed=99)

    assert topo.W.shape == (10, 10)
    # Diagonal is zero
    assert np.allclose(np.diag(topo.W), 0.0)
    # Symmetric
    assert np.allclose(topo.W, topo.W.T, atol=1e-12)
    # Non-negative off-diagonal
    assert np.all(topo.W >= 0.0)
