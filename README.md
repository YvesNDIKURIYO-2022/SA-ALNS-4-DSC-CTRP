# SA-ALNS for DSC-CTRP

**Stability-Aware Adaptive Large Neighborhood Search for the Dynamic Stability-Constrained Container-Tanker Routing Problem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

---

## 📖 Overview

Liquid bulk transportation via multi-compartment tankers presents a critical
operational challenge: balancing economic efficiency with physical safety
throughout dynamic delivery routes. Existing vehicle routing models either
ignore loading constraints entirely or treat them as static
post-optimisation checks, leading to solutions that are economically optimal
but operationally infeasible or unsafe.

This repository provides the complete implementation of the **Stability-Aware
Adaptive Large Neighborhood Search (SA-ALNS)** for solving the **Dynamic
Stability-Constrained Container-Tanker Routing Problem (DSC-CTRP)**.

### Key contributions

| Contribution | Description |
|---|---|
| **Unified formulation** | DSC-CTRP integrating routing decisions with dynamic stability constraints at every delivery prefix |
| **Three-tier constraint hierarchy** | Geometric/capacity, static LDD, and dynamic anti-slosing constraints in one MILP |
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

Wall-clock times are from an idle-host run; penalised costs are deterministic
and identical across reruns. The MILP calls column is uniformly zero because
the LP fallback is never invoked (see *Implementation note* below).

### Key results

- **Against traditional metaheuristics.** SA-ALNS attains 5.70–19.53 % lower
  penalised cost than the four traditional metaheuristics. Significant on both
  paired and unpaired tests against GCOF ($p = 1.9\times10^{-9}$,
  Cohen's $d = -0.844$) and HSA ($p = 9.3\times10^{-9}$, $d = -0.690$). The
  FDAHS and ACO-2opt comparisons are significant on the paired Wilcoxon test
  only and are reported as **marginal** in the manuscript.
- **Against Standard ALNS.** Statistically comparable on cost
  ($p = 0.152$, $d = +0.054$); no claim of improvement is made.
- **Multi-compartment benefit.** Relative to a single-product-per-route
  baseline, the multi-compartment configuration reduces cost by 27.44 %
  ($p = 1.86\times10^{-9}$) and fleet size by 23.68 %
  ($p = 5.32\times10^{-6}$), at the expense of a 34.83 % reduction in volume
  utilisation. The benefit is one of **consolidation**, not loading
  efficiency.
- **Integrated vs. sequential.** Integrated routing and loading outperforms
  sequential decomposition by 37.7 % ($p = 1.9\times10^{-9}$, 30/30
  instances).
- **Ablation nulls.** Removing adaptive weighting, stability-aware repair,
  imbalance destroy, or the feasibility cache changes cost by less than 1 %
  (all $p \geq 0.21$). Operator win rates differ 5.6-fold (destroy) and
  9.8-fold (repair), but the adaptation horizon is too short to exploit the
  difference. This is reported as a **confirmed absence of effect**, not
  suppressed.
- **Safety.** All solutions achieve 100 % centre-of-gravity compliance;
  residual LDD violations are penalised and bounded along the delivery
  sequence.

### Implementation note

The LP fallback in SACA is retained for completeness but **is never invoked
on this benchmark**: the greedy split-allocation procedure succeeds on every
feasible route, so MILP diagnostics are uniformly zero and wall-clock time is
dominated by routing search. This is disclosed explicitly in Sections 4.3.1,
5.3, and 6.4 of the manuscript rather than presenting the LP as a functional
component.

---

## 🔧 Installation

### Prerequisites

- **Python 3.10+**
- **MiKTeX or TeX Live** (optional, for manuscript compilation)
- **`pulp`** (optional, only if you want to exercise `SACA._solve_lp`, which the
  reported benchmark never invokes)

No commercial solver is required. The LP fallback is present as a function
in the script but is never called on the reported benchmark.

### Setup

```bash
git clone https://github.com/YvesNDIKURIYO-2022/SA-ALNS-4-DSC-CTRP.git
cd SA-ALNS-4-DSC-CTRP

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install numpy pandas scipy matplotlib
pip install PyPDF2          # optional: enables figure-integrity verification
```

The four core dependencies are `numpy`, `pandas`, `scipy`, and `matplotlib`.
`PyPDF2` is optional and only affects the final figure-integrity check.

---

## 🚀 Quick Start

The repository is a single script, not a package. Run it from the repository
root:

```bash
python "DSC-CTRP Comprehensive Experimental Comparison.py"
```

The script does the following, in order:

1. Regenerates `benchmark_instances/` if the version marker is missing or
   stale. If the marker is current, it skips regeneration.
2. Runs the seven algorithms on the 30-instance evaluation set, writing
   `results/sota_comparison_results.csv`.
3. Runs the five-seed supplementary comparison, writing
   `results/sota_multiseed.csv` and `results/sota_multiseed_summary.csv`.
4. Runs the five ablation variants, writing
   `results/ablation_study_results.csv`.
5. Runs the 3×3 sensitivity sweep, writing
   `results/sensitivity_analysis_results.csv`.
6. Runs the multi-compartment A/B/C comparison, writing
   `results/multicomp_benefits.csv`.
7. Runs the scalability study on all 100 instances, writing
   `results/sota_scalability.csv`.
8. Writes the figures to `results/part1_sota/` and
   `results/part2_ablation/`.
9. Verifies every emitted PDF for text-layer integrity and reports the
   result.

To use the solver programmatically, load the script as a module:

```python
import importlib.util, json

spec = importlib.util.spec_from_file_location(
    "dsc_ctrp", "DSC-CTRP Comprehensive Experimental Comparison.py")
dsc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsc)

instance = json.load(open("benchmark_instances/P20_1.json"))
vehicle = dsc.create_vehicle_from_instance(instance)
data = dsc.build_instance_data(instance)

solver = dsc.SAALNSAlgorithm(data, vehicle)
solution, cost, extra = solver.solve()

print(f"Penalised cost : {cost:.2f}")
print(f"CG compliance  : {extra['metrics'].cg_compliance_rate:.2f}%")
print(f"LDD violation  : {extra['metrics'].ldd_violation_rate:.2f}%")
print(f"Wall-clock     : {extra['metrics'].routing_time:.3f}s")
```

The script's algorithms, SACA subproblem, and metrics classes are all
available as attributes of the loaded module: `SAALNSAlgorithm`,
`SAALNS_NoStability`, `SAALNS_NoAdaptive`, `SAALNS_Sequential`,
`SAALNS_NoCache`, `SAALNS_NoImbalance`, `SAALNS_SingleProduct`,
`StandardALNS`, `FDAHSAlgorithm`, `GCOFAlgorithm`, `HSAAlgorithm`,
`ACO2optAlgorithm`, `SACA`, `LDDVerifier`, `ObjectiveCalculator`.

---

## 📁 Repository structure

```
SA-ALNS-4-DSC-CTRP/
├── DSC-CTRP Comprehensive Experimental Comparison.py
│       # single entry point: instance generator, SACA, all seven
│       # algorithms, ablation variants, sensitivity sweep,
│       # scalability study, and figure generation
├── LICENSE
└── README.md
```

The script creates `benchmark_instances/` and `results/` on first run. Neither
is committed to the repository: the instances are regenerated from the
embedded generator, and the results are regenerated by the script.

If you add a `manuscript/`, `results/`, or `tests/` directory later, list it
here once it exists. The current listing describes only what the repository
ships.

---

## 📚 Benchmark instances

The study evaluates **100 synthetic benchmark instances** spanning 20–100
customers, 3–5 compartments, and 2–6 products. Of these, **30 instances
(20–25 customers)** are used for the state-of-the-art comparison and the
ablation study reported in the manuscript; the remaining 70 are used for the
scalability study reported in the manuscript's Section 6.4.

| Layer | Instances | Customers | Products | Purpose |
|---|---|---|---|---|
| Paixão layer | 15 | 20–30 | 4 | LDD & stability validation |
| Mirzaei layer | 85 | 20–100 | 2–6 | Scalability testing |
| **Total** | **100** | **20–100** | **2–6** | Full evaluation |

The Paixão layer comprises `P20_1`–`P20_5` (20 customers), `P25_1`–`P25_5`
(25 customers), and `P30_1`–`P30_5` (30 customers). The Mirzaei layer
comprises `vrpnc1`–`vrpnc15` and `extra33`–`extra87`.

All instances are stored in standardised JSON format. Each customer demands
exactly one product, with demand magnitudes calibrated to 60–90 % of the
average compartment volume. The instance generator is embedded in the script;
the instance files are written to `benchmark_instances/` on first run and are
regenerated whenever the version marker changes.

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

A **verified route** passes the sequence-aware feasibility check: cumulative
volume at every delivery prefix ≤ vehicle capacity, and product count ≤
compartment count. Every algorithm produces complete solutions on every
instance, so the verification subset equals the full solution set.

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

The repository was previously published under the identifier
`IALNS-4-DSC-CTRP` and has been renamed to `SA-ALNS-4-DSC-CTRP` for
consistency with the manuscript. GitHub redirects the previous URL to the
current one.

---

## 📬 Contact

| Author | Role | Email |
|---|---|---|
| Yves Ndikuriyo | First author, corresponding author | yvesndikuriyo@csu.edu.cn |
| Yinggui Zhang | Co-author | ygzhang@csu.edu.cn |
| Dung Davou Fom | Co-author | — |

**Affiliation:** School of Traffic and Transportation Engineering, Central
South University, Changsha, Hunan 410075, China

---

## 🙏 Acknowledgments

This work was supported by:

- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province (Grant Nos. 2023JJ30710 and
  2022JJ31020)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE)
file for details.
