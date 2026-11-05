# Deliverables Package Status Report: Game-Theoretic Spectrum Sharing Simulator

**Verification Date & Time:** 2026-10-25 19:40 IST  
**Environment:** macOS (Darwin arm64), Python 3.10.19  
**Core Dependencies:** NumPy 2.2.6, SciPy 1.14.1, Pandas 2.3.3, Matplotlib 3.10.9, PyYAML 6.0.3, Pytest 9.1.1, Tqdm 4.70.1  

---

## 1. Experiment Completion Verification (Audit of Raw CSVs)

Every experiment has **actually completed** and generated un-truncated data across all 100 topologies (seeds 0–99). No experiment is incomplete or missing.

| Experiment | Description | Actual Raw CSV File | Verified Data Rows | Verified Seeds | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Single-topology demo ($N=20, K=4$, seed 42) | `final_deliverables/results/e1_demo_history.csv`<br>`final_deliverables/results/e1_demo_metrics.csv` | 20 rows (history)<br>6 rows (metrics) | Seed 42 | **COMPLETED & VERIFIED** |
| **E2** | Density scaling ($N \in \{10,20,30,40,50\}$, $K=4$) | `final_deliverables/results/e2_density_scaling.csv` | 2,000 data rows (2,001 with header) | Seeds 0 to 99 | **COMPLETED & VERIFIED** |
| **E3** | Channel scaling ($K \in \{2,3,4,5,6,8\}$, $N=20$) | `final_deliverables/results/e3_channel_scaling.csv` | 2,400 data rows (2,401 with header) | Seeds 0 to 99 | **COMPLETED & VERIFIED** |
| **E4** | Optimality gap vs. Exhaustive search ($N=8, K=3$, 6,561 states) | `final_deliverables/results/e4_optimality_gap.csv` | 600 data rows (601 with header) | Seeds 0 to 99 | **COMPLETED & VERIFIED** |
| **E5** | Sensitivity analysis ($n \in \{2.5, 3.5, 4.5\}$, $\sigma_{\text{sh}} \in \{0, 6, 8\}\text{ dB}$) | `final_deliverables/results/e5_sensitivity.csv` | 500 data rows (501 with header) | Seeds 0 to 99 | **COMPLETED & VERIFIED** |
| **E6** | Potential game G1 vs. Rate game G2 comparison | `final_deliverables/results/e6_g2_comparison.csv` | 200 data rows (201 with header) | Seeds 0 to 99 | **COMPLETED & VERIFIED** |

*Verification check executed via Python Pandas on 2026-10-25: confirmed 100% of rows match expected counts and min/max seeds are 0 and 99.*

---

## 2. Test Execution Verification (Pytest: 15 / 15 Passed)

Real execution output captured in [`final_deliverables/tests/pytest_execution_summary.txt`](file:///Users/sakshamjain/Desktop/untitled%20folder/final_deliverables/tests/pytest_execution_summary.txt):

```text
============================= test session starts ==============================
platform darwin -- Python 3.10.19, pytest-9.1.1, pluggy-1.7.0 -- /Users/sakshamjain/Desktop/untitled folder/.venv/bin/python3.10
cachedir: .pytest_cache
rootdir: /Users/sakshamjain/Desktop/untitled folder
configfile: pytest.ini
testpaths: tests
collecting ... collected 15 items

tests/test_baselines.py::test_random_baseline_pair_count_matches_theory PASSED [  6%]
tests/test_baselines.py::test_round_robin_properties PASSED              [ 13%]
tests/test_baselines.py::test_exhaustive_search_finds_global_optimum PASSED [ 20%]
tests/test_channel.py::test_pl0_formula PASSED                           [ 26%]
tests/test_channel.py::test_single_user_sinr_equals_snr PASSED           [ 33%]
tests/test_channel.py::test_noise_power_calculation PASSED               [ 40%]
tests/test_channel.py::test_interference_matrix_symmetry PASSED          [ 46%]
tests/test_game.py::test_k1_no_moves PASSED                              [ 53%]
tests/test_game.py::test_verify_ne_passes_on_100_percent_of_g1_runs PASSED [ 60%]
tests/test_game.py::test_channel_relabel_invariance PASSED               [ 66%]
tests/test_potential.py::test_potential_identity PASSED                  [ 73%]
tests/test_potential.py::test_phi_monotone_non_decreasing PASSED         [ 80%]
tests/test_potential.py::test_brute_force_local_optimum_on_small_instances PASSED [ 86%]
tests/test_reproducibility.py::test_same_seed_gives_identical_results PASSED [ 93%]
tests/test_reproducibility.py::test_jain_index_properties PASSED         [100%]

============================== 15 passed in 0.68s ==============================
```

---

## 3. Inventory of Delivered Artifacts

### 3.1 Figures (`final_deliverables/figures/`, 200 DPI PNG)
1. `fig1_topology_before_after.png` (180 KB) — Spatial link layout before and after NE convergence
2. `fig2_phi_vs_iteration.png` (101 KB) — Monotonic potential $\Phi$ vs switch steps
3. `fig3_sum_rate_vs_iteration.png` (93 KB) — Network sum-rate growth curve
4. `fig4_method_bar_chart.png` (166 KB) — Method bar chart (Sum-Rate, Jain Index, Outage %) with 95% CI
5. `fig5_rate_cdf.png` (137 KB) — Empirical CDF of individual link rates
6. `fig6_sum_rate_jain_vs_n.png` (346 KB) — Sum-Rate and Jain index vs link density $N$
7. `fig7_sum_rate_vs_k.png` (212 KB) — Sum-Rate vs available channels $K$
8. `fig8_rounds_vs_n_boxplot.png` (61 KB) — Convergence rounds vs $N$ boxplot
9. `fig9_channel_load_histogram.png` (75 KB) — Load balancing distribution across channels
10. `fig10_optimality_gap_histogram.png` (107 KB) — Potential gap $\Delta\Phi$ and sum-rate optimality gap histograms
11. `fig11_sensitivity_plot.png` (165 KB) — Sensitivity vs path loss exponent $n$ and shadowing $\sigma_{\text{sh}}$

### 3.2 Summary Tables (`final_deliverables/tables/`, Markdown & CSV)
1. `table1_n20_baselines.md` & `.csv`:
   - Proposed NE: **2,552.29 ± 53.21 Mbps**, Outage **1.5%**, Jain **0.836**, Converged **100.0%**.
   - Greedy: 2,498.45 ± 54.46 Mbps ($p = 4.81 \times 10^{-7}$).
   - Round-Robin: 1,915.99 ± 58.75 Mbps ($p = 3.90 \times 10^{-18}$).
   - Random: 1,844.16 ± 62.84 Mbps ($p = 3.90 \times 10^{-18}$).
2. `table2_optimality_gap.md` & `.csv`:
   - Proposed NE achieves **1,242.96 ± 31.76 Mbps**, with only a **1.54% ± 0.35%** gap from exhaustive global optimum (1,261.92 Mbps).
3. `table3_g1_vs_g2.md` & `.csv`:
   - G1 converges in **100%** of runs (mean 3.68 rounds).
   - G2 converges in only **87%** of runs (13% limit cycles).

### 3.3 Documentation (`final_deliverables/documentation/`)
- `config.yaml`: Exact simulation parameters (B=20 MHz, fc=3.5 GHz, P=23 dBm, NF=7 dB, n=3.5, sigma=6 dB).
- `README.md`: Concise instructions on how to activate venv, run demo, run tests, and run experiments.
- `demo_metadata.json` & `experiments_metadata.json`: Exact Python and package versions recorded during execution.

---

## 4. Readiness for Final Report and PowerPoint Presentation

The dataset is **100% complete and verified**:
- **All experiments (E1 to E6) are complete** with actual numbers in CSV format.
- **All 11 figures are rendered** at 200 DPI with publication styling and error bars.
- **All 3 summary tables are populated** with mean ± 95% CI and Wilcoxon significance tests.
- **Zero missing data or failed tests**.

The project is fully prepared for drafting the final project report and PowerPoint slide deck.
