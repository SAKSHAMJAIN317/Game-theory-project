"""System configuration and constants loader."""

import os
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List
import yaml


@dataclass
class SystemConfig:
    N: int = 20
    L: float = 300.0
    K: int = 4
    B: float = 20.0e6
    f_c: float = 3.5e9
    c: float = 3.0e8
    P_dBm: float = 23.0
    NF_dB: float = 7.0
    n_pathloss: float = 3.5
    sigma_shadowing: float = 6.0
    rx_dist_min: float = 10.0
    rx_dist_max: float = 40.0

    @property
    def P_mW(self) -> float:
        """Transmitter power in linear mW."""
        return 10.0 ** (self.P_dBm / 10.0)

    @property
    def sigma2_dBm(self) -> float:
        """Thermal noise power in dBm: -174 + 10*log10(B) + NF."""
        return -174.0 + 10.0 * math.log10(self.B) + self.NF_dB

    @property
    def sigma2_mW(self) -> float:
        """Thermal noise power in linear mW."""
        return 10.0 ** (self.sigma2_dBm / 10.0)

    @property
    def PL0_dB(self) -> float:
        """Reference path loss at d_0 = 1m: 20*log10(4*pi*1*f_c/c)."""
        return 20.0 * math.log10(4.0 * math.pi * 1.0 * self.f_c / self.c)


@dataclass
class GameConfig:
    tol: float = 1.0e-9
    max_rounds: int = 100


@dataclass
class BaselineConfig:
    random_draws: int = 50


@dataclass
class ExperimentConfig:
    num_topologies: int = 100
    demo_seed: int = 42
    e2_n_values: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    e3_k_values: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8])
    e4_n: int = 8
    e4_k: int = 3
    e5_n_values: List[float] = field(default_factory=lambda: [2.5, 3.5, 4.5])
    e5_sigma_values: List[float] = field(default_factory=lambda: [0.0, 6.0, 8.0])
    e6_run_g2: bool = True


@dataclass
class PathConfig:
    raw_csv: str = "results/raw_csv"
    figures: str = "results/figures"
    tables: str = "results/tables"
    logs: str = "results/logs"


@dataclass
class AppConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    game: GameConfig = field(default_factory=GameConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file, with fallback defaults."""
    if not os.path.exists(config_path):
        # Look in parent or relative directories
        alt_paths = [
            os.path.join("spectrum_game", config_path),
            os.path.join(os.path.dirname(__file__), "..", config_path),
            os.path.join(os.path.dirname(__file__), "..", "..", config_path),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                config_path = p
                break

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    system_data = data.get("system", {})
    game_data = data.get("game", {})
    baselines_data = data.get("baselines", {})
    exp_data = data.get("experiments", {})
    paths_data = data.get("paths", {})

    return AppConfig(
        system=SystemConfig(**system_data),
        game=GameConfig(**game_data),
        baselines=BaselineConfig(**baselines_data),
        experiments=ExperimentConfig(**exp_data),
        paths=PathConfig(**paths_data),
    )
