SAKSHAM JAIN-2427030737   PRIYANSHI SHARMA-2427030746
# Game-Theoretic Spectrum Sharing Simulator for 5G/6G Networks

[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)]()

A high-performance, reproducible Python simulator implementing game-theoretic distributed channel allocation and spectrum sharing for dense 5G/6G wireless networks. The simulator implements an **Exact Potential Game (EPG)** formulation with sequential best-response dynamics, alongside direct rate maximization, baseline allocation strategies, and exhaustive search for optimality gap benchmarking.

---

## 1. Project Overview & Key Results

In next-generation dense wireless networks (e.g., small cells, D2D communications, unlicensed NR-U spectrum sharing), centralized spectrum orchestration imposes prohibitive signaling overhead. This simulator models distributed spectrum sharing as a strategic non-cooperative game where links autonomously select channels to minimize network mutual interference while guaranteeing convergence to a pure-strategy Nash Equilibrium.

### Key Experimental Findings (100 Real Topologies, Seeds 0–99)

- **Pure Nash Equilibrium Convergence**: Game G1 converges in **100% of runs** within an average of **3.68 ± 0.19 rounds** (14.73 ± 0.80 channel switches), verified by an independent unilateral deviation test ($\epsilon = 10^{-9}$).
- **Throughput Gains**: On default topologies ($N=20$ links, $K=4$ channels, $300\text{ m} \times 300\text{ m}$), the Proposed NE achieves **2,552.29 ± 53.21 Mbps** sum-rate, a **38.4% improvement** over Random Allocation ($1,844.16\text{ Mbps}$, $p = 3.90 \times 10^{-18}$) and **33.2% over Round-Robin** ($1,915.99\text{ Mbps}$, $p = 3.90 \times 10^{-18}$).
- **Near-Global Optimality**: Benchmarked against exhaustive search ($3^8 = 6,561$ combinations per topology for $N=8, K=3$), the Proposed NE incurs an average sum-rate optimality gap of only **1.54% ± 0.35%** from the global sum-rate maximum.
- **Fairness & Outage**: The Proposed NE maintains high fairness (Jain's index $0.836 \pm 0.009$) and suppresses link outage ($\text{SINR} < 0\text{ dB}$) down to **1.5%**, compared to **16.0% outage** under random assignment.
- **G1 vs G2 Stability**: While G1 (Exact Potential) converges unconditionally (100%), the individual rate game G2 fails to converge in 13% of topologies due to cyclic best-response dynamics, demonstrating the necessity of the symmetrized potential formulation.

---

## 2. Mathematical Model & Proofs

### System & Radio Model
All internal computations use linear milliwatts ($\text{mW}$) and meters ($\text{m}$):
- **Deployment Area**: $L \times L\text{ m}^2$ (default $L=300\text{ m}$).
- **Links**: $N$ transmitter-receiver pairs. Transmitters $tx_i \sim \mathcal{U}([0, L] \times [0, L])$; receivers $rx_i$ placed at distance $d \sim \mathcal{U}[10, 40]\text{ m}$ with angle $\theta \sim \mathcal{U}[0, 2\pi)$.
- **Radio Parameters**:
  - Channels: $K$ orthogonal subbands (default $K=4$), each with bandwidth $B = 20\text{ MHz}$.
  - Carrier Frequency: $f_c = 3.5\text{ GHz}$, Speed of Light $c = 3.0 \times 10^8\text{ m/s}$.
  - Transmit Power: $P = 23\text{ dBm} = 10^{2.3}\text{ mW} \approx 199.53\text{ mW}$.
  - Receiver Noise Figure: $NF = 7\text{ dB}$.
  - Thermal Noise Power:
    $$\sigma^2_{\text{dBm}} = -174 + 10 \log_{10}(B) + NF = -174 + 73.01 + 7 = -93.99\text{ dBm}$$
    $$\sigma^2 = 10^{\sigma^2_{\text{dBm}} / 10}\text{ mW} \approx 3.99 \times 10^{-10}\text{ mW}$$
- **Path Loss Model**:
  $$PL_0 = 20 \log_{10}\left(\frac{4\pi \cdot 1 \cdot f_c}{c}\right) \approx 43.33\text{ dB}$$
  $$PL_{\text{dB}}(d_{j,i}) = PL_0 + 10 \cdot n \cdot \log_{10}(\max(d_{j,i}, 1)) + X_{j,i}$$
  with path loss exponent $n = 3.5$ and log-normal shadowing $X_{j,i} \sim \mathcal{N}(0, \sigma_{\text{sh}}^2)$ independent per transmitter-receiver pair ($\sigma_{\text{sh}} = 6\text{ dB}$).
- **Linear Channel Gain**:
  $$g[j,i] = 10^{-PL_{\text{dB}}(d_{j,i}) / 10}$$
- **SINR and Shannon Rate**:
  $$\text{SINR}_i(a) = \frac{P \cdot g[i,i]}{\sigma^2 + \sum_{j \neq i, a_j = a_i} P \cdot g[j,i]}$$
  $$R_i(a) = B \log_2(1 + \text{SINR}_i(a))\quad [\text{bps}]$$

---

### Game Formulation G1 (Exact Potential Game)

- **Players**: $\{1, 2, \dots, N\}$ (the $N$ Tx-Rx communication links).
- **Action Space**: $a_i \in \{0, 1, \dots, K-1\}$ for each link $i$.
- **Symmetric Coupling Matrix**:
  $$W[i,j] = \frac{P (g[i,j] + g[j,i])}{\sigma^2} \quad (i \neq j), \quad W[i,i] = 0$$
  Notice that $W[i,j] = W[j,i]$ is symmetric and non-negative.
- **Utility Function**:
  $$u_i(a) = -\sum_{j \neq i} W[i,j] \cdot \mathbb{I}(a_i = a_j)$$
  Maximizing $u_i(a)$ corresponds to minimizing total pairwise interference coupling with co-channel neighbors.
- **Exact Potential Function**:
  $$\Phi(a) = -\sum_{i < j} W[i,j] \cdot \mathbb{I}(a_i = a_j) = -\frac{1}{2} \sum_{i=1}^N \sum_{j \neq i} W[i,j] \cdot \mathbb{I}(a_i = a_j)$$

#### Mathematical Proof of the Exact Potential Property
Suppose player $i$ unilaterally changes channel from $a_i$ to $a'_i$, while all other players retain their actions $a_{-i}$:
1. **Change in Player $i$'s Utility**:
   $$\Delta u_i = u_i(a'_i, a_{-i}) - u_i(a_i, a_{-i}) = \sum_{j \neq i, a_j = a_i} W[i,j] - \sum_{j \neq i, a_j = a'_i} W[i,j]$$
2. **Change in Potential Function**:
   In the sum $\Phi(a) = -\sum_{k < l} W[k,l] \mathbb{I}(a_k = a_l)$, the only terms affected by player $i$'s switch are pairs where either $k = i$ or $l = i$. Since $W$ is symmetric ($W[i,j] = W[j,i]$):
   $$\Delta \Phi = \Phi(a'_i, a_{-i}) - \Phi(a_i, a_{-i}) = \sum_{j \neq i, a_j = a_i} W[i,j] - \sum_{j \neq i, a_j = a'_i} W[i,j]$$
3. **Identity**:
   $$\Delta u_i(a_i \to a'_i, a_{-i}) \equiv \Delta \Phi(a_i \to a'_i, a_{-i}) \quad \forall i, a_i, a'_i, a_{-i}$$
   Therefore, game G1 is an **Exact Potential Game (EPG)**.

#### Convergence Guarantee
Because the action space $K^N$ is finite and every strict improvement in individual utility strictly increases the bounded potential $\Phi(a)$ by at least $\epsilon = 10^{-9}$, sequential best-response dynamics **cannot cycle and must terminate at a pure-strategy Nash Equilibrium in finite steps**.

---

## 3. Directory Layout

```
spectrum_game/
├── config.yaml                     # Simulation parameters and experiment definitions
├── README.md                       # Comprehensive documentation and viva guide
├── pytest.ini                      # Pytest runner configuration
├── src/
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Dataclasses and YAML configuration parser
│   ├── channel.py                  # Topology generator, path loss, SINR, Shannon rate
│   ├── game.py                     # G1 potential dynamics, G2 rate dynamics, verify_NE
│   ├── baselines.py                # Random, Round-Robin, One-Pass Greedy, Exhaustive
│   ├── metrics.py                  # Jain index, INR, outage, Wilcoxon signed-rank test
│   ├── simulator.py                # Multi-experiment runner, orchestrator, table generator
│   └── visualization.py            # Matplotlib 200 DPI publication plotting functions
├── scripts/
│   ├── run_demo.py                 # CLI: Single-topology detailed demonstration (E1)
│   └── run_experiments.py          # CLI: Complete E1-E6 simulation suite runner
├── tests/
│   ├── test_channel.py             # PL0 formula, single-user SINR==SNR, W symmetry
│   ├── test_potential.py           # Potential identity, monotonicity, local optimum
│   ├── test_game.py                # K=1 no moves, 100% NE verification, relabel invariance
│   ├── test_baselines.py           # Random pair count vs theory, round-robin properties
│   └── test_reproducibility.py     # Deterministic bit-for-bit replay on identical seeds
└── results/
    ├── raw_csv/                    # Raw simulation logs and per-run metric CSVs
    ├── figures/                    # 11 publication-grade PNG plots (200 DPI)
    ├── tables/                     # Summary tables in Markdown and CSV formats
    └── logs/                       # Environment metadata, package versions, and logs
```

---

## 4. Quickstart: Installation & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Required libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`, `tqdm`

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib pyyaml pytest tqdm
```

### 1. Run Unit Tests (15 Passed)
```bash
pytest
```
*Executes all mathematical validation tests (potential identity, PL0, monotonicity, NE verification).*

### 2. Run Single-Topology Demo (Experiment E1)
```bash
python scripts/run_demo.py --config config.yaml --seed 42
```
*Outputs detailed switch-by-switch convergence logs, initial vs final potential, sum-rate gains, and generates figures 1, 2, 3, and 5.*

### 3. Run Complete Experiment Suite (Experiments E1–E6)
```bash
python scripts/run_experiments.py --config config.yaml
```
*Executes 100 topologies (seeds 0–99) across all parameter dimensions, performs paired Wilcoxon tests, and generates all 11 figures and 3 summary tables.*

---

## 5. Experimental Results Summary

### Table 1: Baseline Comparison on $N=20, K=4$ (100 Topologies)
*From `results/tables/table1_n20_baselines.md`*

| Method | Sum-Rate (Mbps) | 5th-Pct Rate (Mbps) | Jain Index | Outage (%) | Converged (%) | Wilcoxon p-val |
| --- | --- | --- | --- | --- | --- | --- |
| **Proposed NE (G1)** | **2,552.29 ± 53.21** | **49.93 ± 3.59** | **0.836 ± 0.009** | **1.5% ± 0.6%** | **100.0%** | **1.0000** |
| One-Pass Greedy | 2,498.45 ± 54.46 | 44.00 ± 3.79 | 0.827 ± 0.009 | 2.6% ± 0.8% | 100.0% | $4.81 \times 10^{-7}$ |
| Round-Robin | 1,915.99 ± 58.75 | 13.35 ± 2.46 | 0.710 ± 0.015 | 12.8% ± 1.6% | 100.0% | $3.90 \times 10^{-18}$ |
| Random (50 draws avg) | 1,844.16 ± 62.84 | 10.15 ± 2.14 | 0.684 ± 0.015 | 16.0% ± 1.6% | 100.0% | $3.90 \times 10^{-18}$ |

### Table 2: Optimality Gap Analysis ($N=8, K=3$, 100 Topologies)
*From `results/tables/table2_optimality_gap.md`*

| Method | Sum-Rate (Mbps) | Sum-Rate Gap (%) | Potential Gap ($\Delta\Phi$) |
| --- | --- | --- | --- |
| **Exhaustive (Best Sum-Rate)** | 1,261.92 ± 31.08 | 0.00% ± 0.00% | $8.14 \times 10^{-1}$ |
| **Exhaustive (Best $\Phi$)** | 1,259.45 ± 31.33 | 0.21% ± 0.10% | $0.00 \times 10^{0}$ |
| **Proposed NE (G1)** | **1,242.96 ± 31.76** | **1.54% ± 0.35%** | **$2.28 \times 10^{0}$** |
| One-Pass Greedy | 1,216.65 ± 32.63 | 3.68% ± 0.65% | $8.67 \times 10^{0}$ |
| Round-Robin | 1,068.38 ± 36.34 | 15.40% ± 1.94% | $1.05 \times 10^{3}$ |
| Random (50 draws avg) | 975.90 ± 36.92 | 22.77% ± 2.10% | $4.74 \times 10^{4}$ |

### Table 3: G1 (Potential Game) vs G2 (Direct Rate Game) Dynamics
*From `results/tables/table3_g1_vs_g2.md`*

| Game Formulation | Convergence Rate (%) | Mean Rounds | Mean Switches | Sum-Rate (Mbps) | Jain Index |
| --- | --- | --- | --- | --- | --- |
| **Proposed NE (G1)** | **100.0%** | **3.68 ± 0.19** | **14.73 ± 0.80** | **2,552.29 ± 53.21** | **0.836 ± 0.009** |
| G2 Rate Game | 87.0% | 26.64 ± 6.44 | 159.50 ± 43.07 | 2,554.79 ± 53.04 | 0.835 ± 0.009 |

---

## 6. Generated Figures (in `results/figures/`, 200 DPI)

1. `fig1_topology_before_after.png`: Spatial deployment with links colored by initial vs converged channel assignment.
2. `fig2_phi_vs_iteration.png`: Monotonic progression of potential $\Phi(a)$ across sequential switch steps.
3. `fig3_sum_rate_vs_iteration.png`: Evolution of aggregate network capacity during best response.
4. `fig4_method_bar_chart.png`: Method comparison on Sum-Rate, Jain Fairness, and Outage Fraction with 95% CI error bars.
5. `fig5_rate_cdf.png`: Empirical CDF of user rates highlighting cell-edge performance improvement.
6. `fig6_sum_rate_jain_vs_n.png`: Network density scaling across $N \in \{10, 20, 30, 40, 50\}$.
7. `fig7_sum_rate_vs_k.png`: Spectrum availability scaling across $K \in \{2, 3, 4, 5, 6, 8\}$.
8. `fig8_rounds_vs_n_boxplot.png`: Boxplot showing rapid convergence complexity scaling as $N$ grows.
9. `fig9_channel_load_histogram.png`: Load balancing across the $K$ orthogonal channels.
10. `fig10_optimality_gap_histogram.png`: Distribution of potential difference $\Delta \Phi$ and sum-rate optimality gaps.
11. `fig11_sensitivity_plot.png`: Robustness analysis across path loss exponents $n \in [2.5, 4.5]$ and shadowing $\sigma_{\text{sh}} \in [0, 8]\text{ dB}$.

---

## 7. College Viva & Technical Interview Questions

### Q1: Why is G1 guaranteed to converge, while G2 is not?
**Answer**: G1 is proven to be an **Exact Potential Game (EPG)** because the individual utility function $u_i(a) = -\sum_{j \neq i, a_j = a_i} W[i,j]$ satisfies $\Delta u_i = \Delta \Phi$ for the global potential function $\Phi(a) = -\sum_{i < j, a_i = a_j} W[i,j]$. Since the action space $K^N$ is finite and any unilateral switch strictly increases $\Phi$, cycles are mathematically impossible, guaranteeing finite convergence. In contrast, in G2 where $u_i = R_i(a)$, link $i$'s rate gain does not equal the global sum-rate change (due to asymmetric interference externalities). Consequently, no exact potential exists, leading to limit cycles (oscillations) in 13% of simulation topologies.

### Q2: What is the physical meaning of the matrix $W$?
**Answer**: $W[i,j] = \frac{P (g[i,j] + g[j,i])}{\sigma^2}$. It represents the **symmetrized mutual interference impact** between link $i$ and link $j$ normalized by thermal noise. By symmetrizing $g[i,j]$ (Tx $i \to$ Rx $j$) and $g[j,i]$ (Tx $j \to$ Rx $i$), player $i$ internalizes not only the interference it experiences from player $j$, but also the interference it causes to player $j$. This eliminates the classical "tragedy of the commons" in wireless interference.

### Q3: Why is $\Phi(a)$ negative, and how is the optimality gap defined?
**Answer**: $\Phi(a) = - \sum_{i < j} W[i,j] [a_i == a_j]$ is the negative of the total pairwise interference cost $C(a) = -\Phi(a) \ge 0$. Maximizing $\Phi$ minimizes interference cost. Because $\Phi \le 0$, a standard ratio $\Phi_{\text{NE}} / \Phi^*$ would produce invalid inverted values. Therefore, we evaluate:
1. **Potential Difference**: $\Delta \Phi = \Phi^* - \Phi(a_{\text{NE}}) \ge 0$.
2. **Sum-Rate Optimality Gap**: $\frac{R^*_{\text{sum}} - R_{\text{sum}}(a_{\text{NE}})}{R^*_{\text{sum}}}$.
On 100 topologies, the proposed NE incurs only a **1.54%** average sum-rate gap compared to the global exhaustive search.

### Q4: How is tie-breaking and convergence handled in best-response dynamics?
**Answer**: Each round permutes players randomly. A player only switches channel if the candidate channel improves utility by strictly more than $\epsilon = 1.0 \times 10^{-9}$. If the current channel is tied for maximum utility, the player stays on the current channel. When an entire round completes with zero switches, a pure Nash Equilibrium has been reached, which is independently re-verified by testing all possible unilateral deviations for every link.
