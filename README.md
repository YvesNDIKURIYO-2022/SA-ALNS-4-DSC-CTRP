# Updated README for `SA-ALNS-4-DSC-CTRP`

The README is good but has drifted from the current manuscript in five places. Below is the updated `README.md`, followed by a table of what changed and why.

```markdown
# SA-ALNS for DSC-CTRP

**Stability-Aware Adaptive Large Neighborhood Search for the Dynamic Stability-Constrained Container-Tanker Routing Problem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![CPLEX optional](https://img.shields.io/badge/CPLEX-optional-lightgrey)

---

## 📖 Overview

Liquid bulk transportation via multi-compartment tankers presents a critical operational challenge: balancing economic efficiency with physical safety throughout dynamic delivery routes. Existing vehicle routing models either ignore loading constraints entirely or treat them as static post-optimisation checks, leading to solutions that are economically optimal but operationally infeasible or unsafe.

This repository provides the complete implementation of the **Stability-Aware Adaptive Large Neighborhood Search (SA-ALNS)** for solving the **Dynamic Stability-Constrained Container-Tanker Routing Problem (DSC-CTRP)**.

### Key contributions

| Contribution | Description |
|---|---|
| **Unified formulation** | DSC-CTRP integrating routing decisions with dynamic stability constraints at every delivery prefix |
| **Three-tier constraint hierarchy** | Geometric/capacity, static LDD, and dynamic anti-sloshing constraints in one MILP |
| **SA-ALNS algorithm** | Master–slave metaheuristic with embedded stability verification |
| **Sequence-aware cache** | Feasibility cache keyed on the ordered visit sequence, not the customer set (Proposition 1) |
| **Benchmark suite** | 100 instances; 30 used for the SOTA comparison and ablation in the manuscript |
| **Open null result** | Adaptive operator selection and stability-aware repair show no measurable effect on this benchmark |

---

## 📊 Experimental Results

### State-of-the-art comparison (30 instances, 20–25 customers)

| Algorithm | Avg penalised cost ($) | Fleet | LDD violation (%) | CG compliance (%) | Wall-clock (s) | MILP calls |
|---|---|---|---|---|---|---|
| SA-ALNS | 1430.32 | 4.30 | 35.17 | 100.00 | 0.35 | 0 |
| SA-ALNS Cost-Only | 1435.02 | 4.33 | 36.00 | 100.00 | 0.31 | 0 |
| Standard ALNS | **1416.51** | 4.30 | 30.83 | 100.00 | 0.39 | 0 |
| FDAHS | 1550.84 | 4.20 | 22.61 | 100.00 | 1.16 | 0 |
| GCOF | 1777.46 | 4.37 | 37.89 | 100.00 | 0.18 | 0 |
| HSA | 1620.27 | 4.37 | 32.94 | 100.00 | **0.01** | 0 |
| ACO-2opt | 1516.81 | 4.17 | **22.33** | 100.00 | 0.61 | 0 |

Wall-clock times are from an idle-host run; penalised costs are deterministic and identical across reruns. The `MILP calls` column is uniformly zero because the LP fallback is never invoked (see *Implementation note* below).

### Key results

- **Against traditional metaheuristics.** SA-ALNS attains 5.70–19.53 % lower penalised cost than the four traditional metaheuristics. Significant on both paired and unpaired tests against GCOF ($p = 1.9\times10^{-9}$, Cohen's $d = -0.844$) and HSA ($p = 9.3\times10^{-9}$, $d = -0.690$). The FDAHS and ACO-2opt comparisons are significant on the paired Wilcoxon test only and are reported as **marginal** in the manuscript.
- **Against Standard ALNS.** Statistically comparable on cost ($p = 0.152$, $d = +0.054$); no claim of improvement is made.
- **Multi-compartment benefit.** Relative to a single-product-per-route baseline, the multi-compartment configuration reduces cost by 27.44 % ($p = 1.86\times10^{-9}$) and fleet size by 23.68 % ($p = 5.32\times10^{-6}$), at the expense of a 34.83 % reduction in volume utilisation. The benefit is one of **consolidation**, not loading efficiency.
- **Integrated vs. sequential.** Integrated routing and loading outperforms sequential decomposition by 37.7 % ($p = 1.9\times10^{-9}$, 30/30 instances).
- **Ablation nulls.** Removing adaptive weighting, stability-aware repair, imbalance destroy, or the feasibility cache changes cost by less than 1 % (all $p \geq 0.21$). Operator win rates differ 5.6-fold (destroy) and 9.8-fold (repair), but the adaptation horizon is too short to exploit the difference. This is reported as a **confirmed absence of effect**, not suppressed.
- **Safety.** All solutions achieve 100 % centre-of-gravity compliance; residual LDD violations are penalised and bounded along the delivery sequence.

### Implementation note

The LP fallback in SACA is retained for completeness but **is never invoked on this benchmark**: the greedy split-allocation procedure succeeds on every feasible route, so MILP diagnostics are uniformly zero and wall-clock time is dominated by routing search. This is disclosed explicitly in Sections 4.3.1, 5.3, and 6.4 of the manuscript rather than presenting the LP as a functional component.

---

## 🔧 Installation

### Prerequisites

- **Python 3.10+**
- **IBM ILOG CPLEX 22.1.0+** *(optional — used only by the SACA-MILP fallback, which is not exercised on the reported benchmark)*
- **MiKTeX or TeX Live** (for manuscript compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/SA-ALNS-4-DSC-CTRP.git
cd SA-ALNS-4-DSC-CTRP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
click>=8.0.0
tqdm>=4.62.0
pyyaml>=5.4.0
joblib>=1.1.0
```

---

## 🚀 Quick Start

```python
from src.alns import SAALNS
from src.benchmarks import load_instance

# Load a benchmark instance
instance = load_instance("EL_1")

# Create solver
solver = SAALNS(instance)

# Solve
solution = solver.solve()

# Results
print(f"Total Cost: {solution.cost:.2f}")
print(f"CG Compliance: {solution.cg_compliance:.2f}%")
print(f"LDD Violations: {solution.ldd_violations}")
print(f"Stability Margin: {solution.stability_margin:.2f}%")
print(f"CPU Time: {solution.cpu_time:.3f}s")
```

For backward compatibility, `from src.alns import IALNS` still resolves to `SAALNS`.

---

## 📁 Repository structure

```
SA-ALNS-4-DSC-CTRP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── alns/                    # ALNS master module
│   │   ├── master.py            # Main ALNS loop
│   │   ├── destroy_operators.py # Random, Worst-Cost, Compatibility, Imbalance, Shaw
│   │   └── repair_operators.py  # Greedy, Regret, Stability-Aware, Balance
│   ├── stability/               # SACA slave module
│   │   ├── saca_milp.py         # Exact MILP solver (|R_k| ≤ 9); never invoked on benchmark
│   │   ├── saca_heuristic.py    # Heuristic solver (|R_k| > 9)
│   │   └── ldd_verification.py  # LDD constraint checker
│   ├── cache/                   # Feasibility cache
│   │   └── feasibility_cache.py # LRU cache with sequence-aware keys
│   └── utils/                   # Utilities
│       ├── instance_loader.py   # JSON instance loader
│       └── metrics.py           # Performance metrics
│
├── benchmarks/                  # Benchmark instances
│   ├── paixao/                  # 15 instances (20–30 customers, 3 commodities)
│   │   ├── P20_1--P20_5         # 20 customers
│   │   ├── P25_1--P25_5         # 25 customers
│   │   └── P30_1--P30_5         # 30 customers
│   └── mirzaei/                 # 85 instances (20–100 customers, 2–6 commodities)
│       ├── vrpnc1--vrpnc15      # Mirzaei-layer instances
│       └── extra33--extra87     # Extended instances
│
├── results/                     # Result CSVs
│   ├── sota_comparison_results.csv
│   ├── sota_multiseed.csv
│   ├── sota_multiseed_summary.csv
│   ├── ablation_study_results.csv
│   ├── multicomp_benefits.csv
│   ├── sensitivity_analysis_results.csv
│   └── operator_stats.json
│
├── manuscript/                  # LaTeX manuscript
│   ├── Manuscript-SA-ALNS.tex
│   ├── references.bib
│   └── figures/                 # All figures (PNG, PDF)
│
├── tests/                       # Unit tests
│   ├── test_cache_key_order_sensitivity.py
│   └── test_residual_load_prefix.py
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

---

## 📚 Benchmark instances

The study evaluates **100 synthetic benchmark instances** spanning 20–100 customers, 3–5 compartments, and 2–6 products. Of these, **30 instances (20–25 customers)** are used for the state-of-the-art comparison and the ablation study reported in the manuscript; the remaining 70 are reserved for a scalability study reported in the manuscript's Section 6.4.

| Layer | Instances | Customers | Products | Purpose |
|---|---|---|---|---|
| Paixão layer | 15 | 20–30 | 3 | LDD & stability validation |
| Mirzaei layer | 85 | 20–100 | 2–6 | Scalability testing |
| **Total** | **100** | **20–100** | **2–6** | Full evaluation |

All instances are stored in standardised JSON format. Each customer demands exactly one product, with demand magnitudes calibrated to 60–90 % of the average compartment volume.

---

## 🧠 Algorithm architecture

The SA-ALNS framework follows a master–slave architectural pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Generate Initial                      │
│                   Stability-Aware Solution                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Master Module: ALNS Routing Search                 │
│  ┌────────────────────────┐   ┌──────────────────────────────┐ │
│  │   Destroy Operators    │   │    Repair Operators          │ │
│  │  • Random              │ → │  • Greedy                    │ │
│  │  • Worst-Cost          │   │  • Regret-k                  │ │
│  │  • Compatibility       │   │  • Stability-Aware Insertion │ │
│  │  • Imbalance           │   │  • Balance                   │ │
│  │  • Shaw                │   │                              │ │
│  └────────────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Sequence-Aware Feasibility Cache Check                │
│        (key = ordered customer tuple, not the set)              │
│              HIT ────────────────────────┐                      │
│               MISS                       │                      │
│               ▼                          ▼                      │
│  ┌───────────────────────────┐ ┌────────────────────────────┐   │
│  │  Slave Module: SACA       │ │  Use cached penalty        │   │
│  │  Two-Stage Verification   │ └────────────────────────────┘   │
│  │  Stage 1: Capacity        │                                  │
│  │  & Volume Screening       │                                  │
│  │  Stage 2: Greedy Split-   │                                  │
│  │  Allocation (LP fallback  │                                  │
│  │  retained, not invoked)   │                                  │
│  └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
          ┌─────────────────────────────────────┐
          │       Termination Criterion?        │
          │    YES → Output Final Route Scheme  │
          │    NO → Return to ALNS Main Loop    │
          └─────────────────────────────────────┘
```

---

## 📝 Constraint hierarchy

| Tier | Constraints | Verification type |
|---|---|---|
| 1 | Geometric & capacity | Routing flow, compartment capacity, product compatibility |
| 2 | Static LDD stability | Axle load bounds, centre-of-gravity limits |
| 3 | Dynamic anti-sloshing | Fill-ratio avoidance $[0.35, 0.55]$ + physical validation |

---

## 📊 Performance metrics

| Category | Metrics |
|---|---|
| **Economic efficiency** | Total distance, transportation cost, fleet utilisation, volume utilisation |
| **Safety compliance** | LDD violation rate, CG compliance rate, stability margin, slosh risk score |
| **Computational performance** | CPU time, route-penalty cache hit rate, iterations to convergence, MILP calls |

A **verified route** passes the sequence-aware feasibility check: cumulative volume at every delivery prefix ≤ vehicle capacity, and product count ≤ compartment count. Every algorithm produces complete solutions on every instance, so the verification subset equals the full solution set.

---

## 📄 Citation

If you use this code, data, or benchmarks in your research, please cite:

```bibtex
@article{Ndikuriyo2026,
  title   = {An integrated routing--loading model for dynamic
             stability-constrained multi-compartment tanker routing},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {Computers \& Operations Research},
  year    = {2026},
  note    = {Under review}
}
```

The repository was previously published under the identifier `IALNS-4-DSC-CTRP` and has been renamed to `SA-ALNS-4-DSC-CTRP` for consistency with the manuscript. GitHub redirects the previous URL to the current one.

---

## 📬 Contact

| Author | Role | Email |
|---|---|---|
| Yves Ndikuriyo | First author, corresponding author | yvesndikuriyo@csu.edu.cn |
| Yinggui Zhang | Co-author | ygzhang@csu.edu.cn |
| Dung Davou Fom | Co-author | — |

**Affiliation:** School of Traffic and Transportation Engineering, Central South University, Changsha, Hunan 410075, China

---

## 🙏 Acknowledgments

This work was supported by:

- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province (Grant Nos. 2023JJ30710 and 2022JJ31020)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
