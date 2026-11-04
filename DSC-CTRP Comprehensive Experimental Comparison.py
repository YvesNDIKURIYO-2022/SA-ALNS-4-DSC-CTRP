"""
DSC-CTRP code— standalone reference implementation.

Stability-Aware Adaptive Large Neighborhood Search for the Dynamic
Stability-Constrained Container-Tanker Routing Problem.

This file is self-contained: it generates the benchmark, runs the seven
algorithms, the ablation, the sensitivity sweep, and the scalability
study, and writes the CSVs and figures that back the manuscript.

Dependencies: numpy, pandas, scipy, matplotlib.
The LP fallback in SACA is not included: it is never invoked on the
benchmark, so a solver is not required.

Usage:
    python run_experiments.py --help
    python run_experiments.py --max-instances 30 --seeds 1
    python run_experiments.py --supplementary --format png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

def _resolve_dirs() -> Tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for candidate in (os.path.join(here, "benchmark_instances"),
                      os.path.join(here, "..", "benchmark_instances"),
                      os.path.join(os.getcwd(), "benchmark_instances")):
        if os.path.isdir(candidate):
            base = os.path.abspath(candidate)
            break
    else:
        base = os.path.join(here, "benchmark_instances")
        os.makedirs(base, exist_ok=True)
    out = os.environ.get("DSC_CTRP_OUTPUT") or os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    for sub in ("part1_sota", "part2_ablation", "part3_sensitivity",
                "part4_statistics", "supplementary"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return base, out


BASE_DIR, OUTPUT_DIR = _resolve_dirs()
INSTANCE_GENERATOR_VERSION = "v8.7"

_GLOBAL_SACA_CACHE: Dict = {}


# ============================================================================
# SEED
# ============================================================================

def stable_seed(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16) % (2 ** 31)


# ============================================================================
# STYLE
# ============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 3,
    "ps.fonttype": 3,
    "svg.fonttype": "path",
})

COLORS_SOTA = {
    "SA-ALNS": "#2E86AB", "SA-ALNS_CostOnly": "#7FB3D5",
    "Standard_ALNS": "#1A7A5A", "FDAHS": "#E76F51",
    "GCOF": "#F4A261", "HSA": "#E9C46A", "ACO_2opt": "#2A9D8F",
}
SOTA_DISPLAY_NAMES = {
    "SA-ALNS": "SA-ALNS (Proposed)",
    "SA-ALNS_CostOnly": "SA-ALNS (Cost-Only Ablation)",
    "Standard_ALNS": "Standard ALNS",
    "FDAHS": "FDAHS (2026)", "GCOF": "GCOF (2026)",
    "HSA": "HSA (2023)", "ACO_2opt": "ACO-2opt (2015)",
}
MARKERS = {
    "SA-ALNS": "o", "SA-ALNS_CostOnly": "X", "Standard_ALNS": "s",
    "FDAHS": "^", "GCOF": "D", "HSA": "v", "ACO_2opt": "p",
    "SA-ALNS_NS": "*", "SA-ALNS_NA": "h", "SA-ALNS_SEQ": "<",
    "SA-ALNS_NC": ">", "SA-ALNS_IMB": "P",
}
COLORS_ABLATION = {
    "SA-ALNS": "#2E86AB", "SA-ALNS_NS": "#F18F01", "SA-ALNS_NA": "#C73E1D",
    "SA-ALNS_SEQ": "#73AB84", "SA-ALNS_NC": "#6A4C93", "SA-ALNS_IMB": "#D4A373",
}
ABLATION_DISPLAY_NAMES = {
    "SA-ALNS": "SA-ALNS (Full)", "SA-ALNS_NS": "No Stability",
    "SA-ALNS_NA": "No Adaptive", "SA-ALNS_SEQ": "Sequential",
    "SA-ALNS_NC": "No Cache", "SA-ALNS_IMB": "No Imbalance",
}


# ============================================================================
# SAFE SAVE
# ============================================================================

def verify_pdf_integrity(path: str) -> bool:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return True
    try:
        reader = PdfReader(path)
        if not reader.pages:
            return False
        text = "".join((p.extract_text() or "") for p in reader.pages).strip()
        if not text:
            return False
        chars = [c for c in text if not c.isspace()]
        if not chars or len(set(chars)) < 10:
            return False
        if re.search(r"text\[\[|image\[\[|<table>", text):
            return False
        return True
    except Exception:
        return False


def safe_savefig(fig, path: str, retries: int = 3, sleep: float = 1.0,
                 save_format: Optional[str] = None) -> Optional[str]:
    if save_format is None:
        save_format = "png" if path.endswith(".png") else "pdf"
    for attempt in range(retries):
        try:
            fig.savefig(path, format=save_format)
            if save_format == "pdf" and not verify_pdf_integrity(path):
                png_path = path[:-4] + ".png" if path.endswith(".pdf") else path + ".png"
                fig.savefig(png_path, format="png")
                if verify_pdf_integrity(png_path):
                    print(f"  [warn] {path} corrupted; wrote {png_path}")
                    return png_path
                raise RuntimeError("both PDF and PNG corrupted")
            return path
        except PermissionError:
            time.sleep(sleep)
        except Exception as e:
            print(f"  [warn] attempt {attempt + 1} for {path}: {e}")
            time.sleep(sleep)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    base, ext = os.path.splitext(path)
    alt = f"{base}_{int(time.time())}{ext}"
    try:
        fig.savefig(alt, format=save_format)
        print(f"  [warn] {path} locked; wrote {alt}")
        return alt
    except Exception as e:
        print(f"  [error] could not save {path}: {e}")
        return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Compartment:
    id: int
    capacity: float
    position_x: float
    position_y: float
    load_density: float


@dataclass
class Vehicle:
    id: int
    compartments: List[Compartment]
    max_payload: float
    front_axle_max: float
    rear_axle_max: float
    wheelbase: float
    unladen_weight: float
    max_volume: float
    steering_min_ratio: float
    driving_min_ratio: float
    product_densities: Dict[int, float] = field(default_factory=dict)
    fixed_cost: float = 100.0
    cost_per_km: float = 1.0
    front_axle_min: float = 0.0
    rear_axle_min: float = 0.0
    cg_min: float = 0.0
    cg_max: float = 0.0
    compartment_positions: Dict[int, float] = field(default_factory=dict)
    is_multi_compartment: bool = True


@dataclass
class LDDResult:
    feasible: bool
    front_load: float
    rear_load: float
    center_of_gravity: float
    stability_margin: float
    violations: List[str]
    cg_position: float = 0.0
    cg_min: float = 0.0
    cg_max: float = 0.0
    cg_compliant: bool = True
    total_mass: float = 0.0
    longitudinal_moment: float = 0.0


@dataclass
class Metrics:
    total_distance: float = 0.0
    transportation_cost: float = 0.0
    fleet_size: int = 0
    fleet_utilization: float = 0.0
    volume_utilization: float = 0.0
    ldd_violation_rate: float = 0.0
    stability_margin: float = 0.0
    minimum_load_ratio: float = 0.0
    slosh_risk_score: float = 0.0
    cpu_time: float = 0.0
    saca_calls: int = 0
    cache_hit_rate: float = 0.0
    iterations_to_convergence: int = 0
    milp_calls: int = 0
    cache_hits: int = 0
    milp_time: float = 0.0
    routing_time: float = 0.0
    feasibility_failures: int = 0
    feasibility_attempts: int = 0
    cg_position_mean: float = 0.0
    cg_position_std: float = 0.0
    cg_position_min: float = 0.0
    cg_position_max: float = 0.0
    cg_compliance_rate: float = 100.0
    cg_deviation_from_optimum: float = 0.0
    heeling_moment_estimate: float = 0.0
    lateral_acceleration_margin: float = 0.0
    penalised_cost: float = 0.0
    avg_penalty: float = 0.0
    n_total_routes: int = 0
    n_verified_routes: int = 0
    route_penalty_cache_hits: int = 0
    route_penalty_cache_hit_rate: float = 0.0
    op_stats: Optional[Dict] = None


@dataclass
class SensitivityResult:
    lambda_W: float
    lambda_S: float
    avg_cost: float
    avg_stability: float
    avg_ldd_violation: float
    avg_fleet_size: float
    avg_cg_compliance: float
    avg_cpu_time: float
    std_cost: float
    std_stability: float
    convergence_iterations: int
    parameter_combination_id: str


# ============================================================================
# LDD VERIFIER
# ============================================================================

class LDDVerifier:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.positions = {
            c.id: float(np.clip(c.position_x, 0.0, vehicle.wheelbase))
            for c in vehicle.compartments
        }
        self.front_min = getattr(vehicle, "front_axle_min", 0.0)
        self.rear_min = getattr(vehicle, "rear_axle_min", 0.0)
        self.cg_min = getattr(vehicle, "cg_min", 0.3 * vehicle.wheelbase)
        self.cg_max = getattr(vehicle, "cg_max", 0.7 * vehicle.wheelbase)

    def verify(self, loads: Dict[int, float]) -> LDDResult:
        active = {k: v for k, v in loads.items() if v > 1e-9}
        weight = sum(active.values())
        wheelbase = self.vehicle.wheelbase

        if weight <= 1e-9:
            unladen = self.vehicle.unladen_weight
            cg = 0.5 * wheelbase
            rear = unladen * cg / wheelbase
            front = unladen - rear
            violations = []
            if front > self.vehicle.front_axle_max:
                violations.append("front_overload")
            if rear > self.vehicle.rear_axle_max:
                violations.append("rear_overload")
            return LDDResult(True if not violations else False, front, rear, cg,
                             100.0, violations, cg, self.cg_min, self.cg_max,
                             True, 0.0, 0.0)

        moment = sum(load * self.positions.get(cid, 0.5 * wheelbase)
                     for cid, load in active.items())
        cargo_cg = moment / weight
        unladen = self.vehicle.unladen_weight
        unladen_cg = 0.5 * wheelbase
        total_mass = weight + unladen
        cg = (weight * cargo_cg + unladen * unladen_cg) / total_mass
        rear = total_mass * cg / wheelbase
        front = total_mass - rear

        violations = []
        if front > self.vehicle.front_axle_max:
            violations.append("front_overload")
        if front < self.front_min:
            violations.append("front_underload")
        if rear > self.vehicle.rear_axle_max:
            violations.append("rear_overload")
        if rear < self.rear_min:
            violations.append("rear_underload")
        if weight > self.vehicle.max_payload:
            violations.append("payload")

        cg_ok = True
        if cg < self.cg_min:
            violations.append("cg_forward")
            cg_ok = False
        if cg > self.cg_max:
            violations.append("cg_rearward")
            cg_ok = False

        if total_mass > 0:
            if front / total_mass < self.vehicle.steering_min_ratio:
                violations.append("steering")
            if rear / total_mass < self.vehicle.driving_min_ratio:
                violations.append("driving")

        margins = []
        if self.vehicle.front_axle_max > 0:
            margins.append((self.vehicle.front_axle_max - front) / self.vehicle.front_axle_max)
        if self.vehicle.rear_axle_max > 0:
            margins.append((self.vehicle.rear_axle_max - rear) / self.vehicle.rear_axle_max)
        if self.vehicle.max_payload > 0:
            margins.append((self.vehicle.max_payload - weight) / self.vehicle.max_payload)
        if self.cg_max > self.cg_min:
            margins.append((cg - self.cg_min) / (self.cg_max - self.cg_min))
            margins.append((self.cg_max - cg) / (self.cg_max - self.cg_min))
        margin = max(0.0, min(margins)) * 100 if margins else 100.0

        return LDDResult(
            feasible=not violations,
            front_load=front, rear_load=rear, center_of_gravity=cg,
            stability_margin=margin, violations=violations,
            cg_position=cg, cg_min=self.cg_min, cg_max=self.cg_max,
            cg_compliant=cg_ok, total_mass=weight,
            longitudinal_moment=moment,
        )


# ============================================================================
# SACA
# ============================================================================

class SACA:
    RISKY_BAND = (0.35, 0.55)
    W_AXLE = 100.0
    W_CG = 500.0
    W_SLOSH = 2.0
    SLOSH_CAP_FRACTION = 0.20

    def __init__(self, vehicle: Vehicle, instance_data: Optional[Dict] = None,
                 stability_weight: float = 0.35, use_cache: bool = True):
        self.vehicle = vehicle
        self.instance_data = instance_data
        self.ldd = LDDVerifier(vehicle)
        self.stability_weight = stability_weight
        self.sig = (round(vehicle.wheelbase, 4), round(vehicle.max_payload, 3),
                    tuple(round(c.capacity, 3) for c in vehicle.compartments),
                    tuple(round(c.position_x, 3) for c in vehicle.compartments))
        self.use_cache = use_cache
        self.cache = _GLOBAL_SACA_CACHE if use_cache else {}
        self.milp_calls = 0
        self.cache_hits = 0
        self.milp_time = 0.0
        self.failure_reasons: Dict[str, int] = defaultdict(int)

    @staticmethod
    def key(route: List[int]) -> str:
        return hashlib.sha1(repr(tuple(route)).encode()).hexdigest()

    def _sequence_feasible(self, route, demands) -> bool:
        k = len(route)
        if k == 0:
            return True
        suffix = [dict() for _ in range(k)]
        running: Dict[int, float] = defaultdict(float)
        for i in range(k - 1, -1, -1):
            for p, v in demands[route[i]].items():
                running[p] += v
            suffix[i] = dict(running)
        for remaining in suffix:
            if sum(remaining.values()) > self.vehicle.max_volume + 1e-6:
                return False
            if sum(1 for v in remaining.values() if v > 1e-6) > len(self.vehicle.compartments):
                return False
        return True

    def verify(self, route, demands, transport_cost=None,
               skip_sequence_check=False):
        if not route:
            return 0.0, {}, {}, False

        cache_key = (self.sig, SACA.key(route))
        if self.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        if not skip_sequence_check and not self._sequence_feasible(route, demands):
            self.failure_reasons["sequence_infeasible"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[cache_key] = res
            return res

        totals: Dict[int, float] = defaultdict(float)
        for c in route:
            for p, v in demands[c].items():
                totals[p] += v

        if sum(totals.values()) > self.vehicle.max_volume + 1e-6:
            self.failure_reasons["total_volume"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[cache_key] = res
            return res

        if sum(1 for v in totals.values() if v > 1e-6) > len(self.vehicle.compartments):
            self.failure_reasons["too_many_products"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[cache_key] = res
            return res

        vol_alloc, prod_alloc = self._greedy_alloc(totals)
        if not vol_alloc:
            self.failure_reasons["allocation"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[cache_key] = res
            return res

        penalty = self._penalty(prod_alloc, vol_alloc, transport_cost)
        res = (penalty, vol_alloc, prod_alloc, False)
        if self.use_cache:
            self.cache[cache_key] = res
        return res

    def _greedy_alloc(self, demand: Dict[int, float]):
        comps = self.vehicle.compartments
        products = list(demand.keys())
        if not products or sum(demand.values()) <= 0:
            return {}, {}
        if sum(1 for v in demand.values() if v > 1e-6) > len(comps):
            return {}, {}

        ordered = sorted(demand.items(), key=lambda x: x[1], reverse=True)
        state = {c.id: {"alloc": {}, "total": 0.0} for c in comps}

        for product_id, vol in ordered:
            need = vol
            for comp in comps:
                if need <= 1e-6:
                    break
                st = state[comp.id]
                if product_id not in st["alloc"]:
                    continue
                avail = comp.capacity - st["total"]
                if avail <= 1e-6:
                    continue
                take = min(need, avail)
                st["alloc"][product_id] += take
                st["total"] += take
                need -= take
            for comp in comps:
                if need <= 1e-6:
                    break
                st = state[comp.id]
                if st["total"] > 1e-6:
                    continue
                take = min(need, comp.capacity)
                st["alloc"][product_id] = take
                st["total"] = take
                need -= take
            for comp in comps:
                if need <= 1e-6:
                    break
                st = state[comp.id]
                if product_id in st["alloc"]:
                    continue
                avail = comp.capacity - st["total"]
                if avail <= 1e-6:
                    continue
                take = min(need, avail)
                st["alloc"][product_id] = take
                st["total"] += take
                need -= take
            if need > 1e-3:
                return {}, {}

        vol_alloc, prod_alloc = {}, {}
        for cid, st in state.items():
            if st["total"] <= 1e-6:
                vol_alloc[cid] = 0.0
                prod_alloc[cid] = {}
            else:
                vol_alloc[cid] = st["total"]
                prod_alloc[cid] = dict(st["alloc"])
        if abs(sum(vol_alloc.values()) - sum(demand.values())) > 1e-3:
            return {}, {}
        return vol_alloc, prod_alloc

    def _penalty(self, prod_alloc, vol_alloc, transport_cost) -> float:
        loads = {cid: vol * 0.8 for cid, vol in vol_alloc.items() if vol > 1e-6}
        if not loads:
            return 0.0

        ldd = self.ldd.verify(loads)
        penalty = 0.0
        for v in ldd.violations:
            if v == "front_overload":
                penalty += self.W_AXLE * (ldd.front_load - self.vehicle.front_axle_max)
            elif v == "rear_overload":
                penalty += self.W_AXLE * (ldd.rear_load - self.vehicle.rear_axle_max)
            elif v == "payload":
                penalty += self.W_AXLE * (ldd.total_mass - self.vehicle.max_payload)
            elif v == "cg_forward":
                penalty += self.W_CG * (self.vehicle.cg_min - ldd.cg_position)
            elif v == "cg_rearward":
                penalty += self.W_CG * (ldd.cg_position - self.vehicle.cg_max)

        slosh = 0.0
        for cid, kg in loads.items():
            comp = next((c for c in self.vehicle.compartments if c.id == cid), None)
            if comp is None or comp.capacity <= 0:
                continue
            fill = (kg / 0.8) / comp.capacity
            if self.RISKY_BAND[0] <= fill <= self.RISKY_BAND[1]:
                slosh += self.W_SLOSH
            elif 0.25 <= fill < self.RISKY_BAND[0] or self.RISKY_BAND[1] < fill <= 0.65:
                slosh += 0.5 * self.W_SLOSH

        if transport_cost is not None and transport_cost > 0:
            slosh = min(slosh, self.SLOSH_CAP_FRACTION * transport_cost)
        return penalty + slosh


class SACASingleProduct(SACA):
    def verify(self, route, demands, transport_cost=None,
               skip_sequence_check=False):
        if not route:
            return 0.0, {}, {}, False
        totals: Dict[int, float] = defaultdict(float)
        for c in route:
            for p, v in demands[c].items():
                totals[p] += v
        if sum(1 for v in totals.values() if v > 1e-6) > 1:
            self.failure_reasons["multi_product_route"] += 1
            return (1e6, {}, {}, True)
        return super().verify(route, demands, transport_cost, skip_sequence_check)


# ============================================================================
# OBJECTIVE
# ============================================================================

class Objective:
    def __init__(self, instance_data, vehicle, saca=None, stability_weight=0.35):
        self.data = instance_data
        self.vehicle = vehicle
        self.ldd = LDDVerifier(vehicle)
        self.saca = saca or SACA(vehicle, instance_data, stability_weight=stability_weight)
        self.dm = instance_data["distance_matrix"]

    def distance(self, solution) -> float:
        total = 0.0
        for route in solution.values():
            if not route:
                continue
            nodes = [0] + [c + 1 for c in route] + [0]
            total += sum(self.dm[nodes[i]][nodes[i + 1]] for i in range(len(nodes) - 1))
        return total

    def cost(self, solution) -> float:
        return self.vehicle.cost_per_km * self.distance(solution) + self.vehicle.fixed_cost * len(solution)

    def _verify_route(self, route):
        demands = {c: self.data["customers"][c]["demand"] for c in route}
        return self.saca.verify(route, demands,
                                transport_cost=self.cost({0: route}))

    def penalised_cost(self, solution, lambda_W=1.0, lambda_S=0.5) -> float:
        travel = self.cost(solution)
        total_penalty = 0.0
        for route in solution.values():
            if not route:
                continue
            penalty, _, _, hard = self._verify_route(route)
            if hard:
                return float("inf")
            total_penalty += penalty
        return travel + (lambda_W + lambda_S) * total_penalty

    def metrics(self, solution, iterations=0, routing_time=0.0,
                lambda_W=1.0, lambda_S=0.5) -> Metrics:
        start = time.time()
        verified = []
        for route in solution.values():
            if not route:
                continue
            verified.append((route, self._verify_route(route)))
        n_total = len(verified)
        n_verified = sum(1 for _, (_, _, _, hard) in verified if not hard)

        ldd_count = 0
        stab, cgs, loads_ratios = [], [], []
        compliant = 0
        for route, (_, vol_alloc, _, hard) in verified:
            if hard:
                continue
            loads = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
            ldd = self.ldd.verify(loads)
            if ldd.cg_compliant:
                compliant += 1
            if ldd.feasible:
                stab.append(ldd.stability_margin)
                cgs.append(ldd.cg_position)
                if ldd.total_mass > 0:
                    total_mass = ldd.total_mass + self.vehicle.unladen_weight
                    if total_mass > 0:
                        loads_ratios.append(ldd.front_load / total_mass * 100)
            if self._verify_route(route)[0] > 1e-6:
                ldd_count += 1

        avg_penalty = 0.0
        slosh = 0.0
        n_pen = 0
        for route, (penalty, _, _, hard) in verified:
            if not hard:
                avg_penalty += penalty
                slosh += penalty
                n_pen += 1
        avg_penalty /= max(n_pen, 1)
        slosh /= max(n_pen, 1)

        total_vol = sum(sum(self._demands(r).values()) for r in solution.values() if r)
        total_cap = self.vehicle.max_volume * sum(1 for r in solution.values() if r)
        vol_util = (total_vol / total_cap * 100) if total_cap > 0 else 0.0

        fleet_util = 0.0
        if solution:
            util = sum(1 for r in solution.values()
                       if r and sum(self._demands(r).values()) / max(self.vehicle.max_volume, 1e-9) >= 0.8)
            fleet_util = util / len(solution) * 100

        hm = 0.0
        for route, (_, vol_alloc, _, hard) in verified:
            if hard or not vol_alloc:
                continue
            loads = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
            total_mass = sum(loads.values())
            cg_h = 0.5 + float(np.mean([self.ldd.positions.get(cid, 0) for cid in loads]))
            hm = max(hm, total_mass * 0.4 * cg_h)

        wall = time.time() - start
        return Metrics(
            total_distance=self.distance(solution),
            transportation_cost=self.cost(solution),
            fleet_size=len(solution),
            fleet_utilization=fleet_util,
            volume_utilization=vol_util,
            ldd_violation_rate=(ldd_count / n_verified * 100) if n_verified else 0.0,
            stability_margin=float(np.mean(stab)) if stab else 0.0,
            minimum_load_ratio=float(np.mean(loads_ratios)) if loads_ratios else 0.0,
            slosh_risk_score=slosh,
            cpu_time=routing_time + self.saca.milp_time + wall,
            saca_calls=self.saca.milp_calls,
            cache_hit_rate=(self.saca.cache_hits /
                            max(1, self.saca.milp_calls + self.saca.cache_hits)) * 100,
            iterations_to_convergence=iterations,
            milp_calls=self.saca.milp_calls,
            cache_hits=self.saca.cache_hits,
            milp_time=self.saca.milp_time,
            routing_time=routing_time,
            cg_position_mean=float(np.mean(cgs)) if cgs else 0.0,
            cg_position_std=float(np.std(cgs)) if len(cgs) > 1 else 0.0,
            cg_position_min=float(min(cgs)) if cgs else 0.0,
            cg_position_max=float(max(cgs)) if cgs else 0.0,
            cg_compliance_rate=(100.0 * compliant / n_verified) if n_verified else 0.0,
            cg_deviation_from_optimum=abs(float(np.mean(cgs)) - 0.5 * self.vehicle.wheelbase) if cgs else 0.0,
            heeling_moment_estimate=hm,
            lateral_acceleration_margin=(1000 - hm) / 1000 * 100 if hm > 0 else 100.0,
            penalised_cost=self.penalised_cost(solution, lambda_W, lambda_S),
            avg_penalty=avg_penalty,
            n_total_routes=n_total,
            n_verified_routes=n_verified,
        )

    def _demands(self, route) -> Dict[int, float]:
        out: Dict[int, float] = defaultdict(float)
        for cid in route:
            for p, v in self.data["customers"][cid]["demand"].items():
                out[p] += v
        return dict(out)


# ============================================================================
# BENCHMARK
# ============================================================================

def generate_instance(n_customers: int, layer: str, seed: int,
                      name: Optional[str] = None,
                      n_products: int = 4, n_compartments: int = 3) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    name = name or f"{layer}_{n_customers}"
    n_products = max(2, min(n_products, n_compartments + 2))
    wheelbase = random.uniform(3.0, 5.0)

    compartments, positions = [], {}
    for i in range(n_compartments):
        frac = 0.30 + 0.40 * (i + 0.5) / n_compartments
        px = wheelbase * frac
        positions[i] = px
        compartments.append({"id": i, "capacity": random.uniform(120.0, 200.0),
                             "position_x": px, "position_y": 0.0, "load_density": 1.0})

    total_cap = sum(c["capacity"] for c in compartments)
    max_payload = total_cap * 0.8
    unladen = max_payload * random.uniform(0.30, 0.50)
    gvw = max_payload + unladen

    avg_cap = total_cap / n_compartments
    dmax = avg_cap * random.uniform(0.60, 0.90)
    dmin = dmax * 0.40

    customers = []
    for i in range(n_customers):
        p = random.randint(0, n_products - 1)
        customers.append({
            "id": i, "x": random.uniform(0, 100), "y": random.uniform(0, 100),
            "demand": {p: random.uniform(dmin, dmax)},
            "time_window": [0.0, 1440.0], "service_time": random.uniform(5, 15),
        })
    tws = {}
    for i in range(n_customers):
        s = random.uniform(0, 100)
        tws[i] = [s, s + random.uniform(20, 60)]
        customers[i]["time_window"] = tws[i]

    vehicle = {
        "id": 0, "compartments": compartments, "compartment_positions": positions,
        "max_payload": max_payload,
        "front_axle_max": gvw * 0.50, "rear_axle_max": gvw * 0.65,
        "front_axle_min": gvw * 0.05, "rear_axle_min": gvw * 0.065,
        "wheelbase": wheelbase, "unladen_weight": unladen, "max_volume": total_cap,
        "steering_min_ratio": 0.25, "driving_min_ratio": 0.55,
        "product_densities": {p: 0.8 for p in range(n_products)},
        "cg_min": 0.3 * wheelbase, "cg_max": 0.7 * wheelbase,
    }

    coords = [[0, 0]] + [[c["x"], c["y"]] for c in customers]
    dm = [[math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
           for j in range(n_customers + 1)] for i in range(n_customers + 1)]

    return {"name": name, "layer": layer, "customers": customers,
            "time_windows": tws, "vehicle": vehicle, "distance_matrix": dm,
            "n_products": n_products, "n_customers": n_customers,
            "n_compartments": n_compartments}


def ensure_instances(base_dir: str, num_instances: int = 100) -> None:
    marker = os.path.join(base_dir, ".instance_generator_version")
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                if f.read().strip() == INSTANCE_GENERATOR_VERSION:
                    js = [f for _, _, fs in os.walk(base_dir) for f in fs if f.endswith(".json")]
                    if js:
                        print(f"Instances current ({INSTANCE_GENERATOR_VERSION}); {len(js)} files.")
                        return
        except Exception:
            pass

    print(f"\nRegenerating instances with {INSTANCE_GENERATOR_VERSION} ...")
    for root, _, files in os.walk(base_dir):
        if "results" in root:
            continue
        for f in files:
            if f.endswith(".json"):
                os.remove(os.path.join(root, f))

    all_instances = []
    paixao = [("P20", 20, 4, 3), ("P25", 25, 4, 3), ("P30", 30, 4, 4)]
    mirzaei = [(40, 4, 3), (40, 5, 3), (40, 4, 4), (50, 4, 3), (50, 5, 3),
               (50, 5, 4), (60, 5, 3), (60, 5, 4), (60, 4, 4), (75, 5, 4),
               (75, 5, 4), (75, 4, 5), (100, 5, 4), (100, 5, 5), (100, 4, 5)]
    ctr = 1
    for prefix, n, npr, ncmp in paixao:
        for i in range(1, 6):
            nm = f"{prefix}_{i}"
            all_instances.append((nm, generate_instance(n, "Paixao", ctr, nm, npr, ncmp)))
            ctr += 1
    for idx, (n, npr, ncmp) in enumerate(mirzaei, start=1):
        nm = f"vrpnc{idx}"
        all_instances.append((nm, generate_instance(n, "Mirzaei", ctr, nm, npr, ncmp)))
        ctr += 1
    while len(all_instances) < num_instances:
        nm = f"extra{len(all_instances) + 1}"
        all_instances.append((nm, generate_instance(
            random.choice([20, 25, 30, 40, 50, 60, 75, 100]), "Mirzaei", ctr, nm,
            random.choice([3, 4, 5]), random.choice([3, 4, 5]))))
        ctr += 1
    for nm, inst in all_instances:
        with open(os.path.join(base_dir, f"{nm}.json"), "w") as f:
            json.dump(inst, f, indent=2)
    with open(marker, "w") as f:
        f.write(INSTANCE_GENERATOR_VERSION)
    print(f"Generated {len(all_instances)} instances.")


def load_vehicle(instance: Dict, multi_compartment: bool = True) -> Vehicle:
    vd = instance["vehicle"]
    raw = vd["compartments"]
    if multi_compartment:
        comps = [Compartment(c["id"], c["capacity"], c["position_x"],
                             c["position_y"], 1.0) for c in raw]
        positions = {c["id"]: c["position_x"] for c in raw}
        total_cap = sum(c.capacity for c in comps)
    else:
        total_cap = sum(c["capacity"] for c in raw)
        weighted = sum(c["capacity"] * c["position_x"] for c in raw) / max(total_cap, 1e-9)
        comps = [Compartment(0, total_cap, weighted, 0.0, 1.0)]
        positions = {0: weighted}
    return Vehicle(
        id=0, compartments=comps, max_payload=vd["max_payload"],
        front_axle_max=vd["front_axle_max"], rear_axle_max=vd["rear_axle_max"],
        wheelbase=vd["wheelbase"], unladen_weight=vd["unladen_weight"],
        max_volume=total_cap, steering_min_ratio=vd["steering_min_ratio"],
        driving_min_ratio=vd["driving_min_ratio"],
        product_densities=vd.get("product_densities", {}),
        fixed_cost=100.0, cost_per_km=1.0,
        front_axle_min=vd.get("front_axle_min", vd["front_axle_max"] * 0.1),
        rear_axle_min=vd.get("rear_axle_min", vd["rear_axle_max"] * 0.1),
        cg_min=vd.get("cg_min", 0.3 * vd["wheelbase"]),
        cg_max=vd.get("cg_max", 0.7 * vd["wheelbase"]),
        compartment_positions=positions,
        is_multi_compartment=multi_compartment)


def build_data(instance: Dict) -> Dict:
    n = len(instance["customers"])
    coords = [[0, 0]] + [[c["x"], c["y"]] for c in instance["customers"]]
    dm = [[math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
           for j in range(n + 1)] for i in range(n + 1)]
    return {"name": instance["name"], "customers": instance["customers"],
            "time_windows": instance["time_windows"], "distance_matrix": dm,
            "vehicle": instance["vehicle"],
            "n_products": instance.get("n_products", 4), "n_customers": n,
            "n_compartments": instance.get("n_compartments",
                                            len(instance["vehicle"]["compartments"]))}


# ============================================================================
# BASE ALGORITHM
# ============================================================================

class BaseAlgorithm:
    def __init__(self, data, vehicle, stability_weight=0.35):
        self.data = data
        self.vehicle = vehicle
        self.stability_weight = stability_weight
        self.saca = SACA(vehicle, data, stability_weight=stability_weight)
        self.obj = Objective(data, vehicle, saca=self.saca,
                             stability_weight=stability_weight)
        self.dm = data["distance_matrix"]
        self.n = len(data["customers"])
        self.cache: Dict[str, Tuple[float, bool]] = {}
        self.cache_hits = 0
        self.attempts = 0
        self.failures = 0
        self.lambda_W = 1.0
        self.lambda_S = 0.5
        self.history: List[float] = []

    def _complete(self, solution) -> bool:
        seen = set()
        for route in solution.values():
            for c in route:
                if c in seen:
                    return False
                seen.add(c)
        return len(seen) == self.n

    def _route_penalty(self, route) -> Tuple[float, bool]:
        if not route:
            return 0.0, False
        key = SACA.key(route)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.attempts += 1
        demands = {c: self.data["customers"][c]["demand"] for c in route}
        if not self.saca._sequence_feasible(route, demands):
            self.cache[key] = (1e6, True)
            self.failures += 1
            return 1e6, True
        travel = self.obj.cost({0: route})
        penalty, _, _, hard = self.saca.verify(route, demands,
                                              transport_cost=travel,
                                              skip_sequence_check=True)
        self.cache[key] = (penalty, hard)
        if hard or penalty > 1e-6:
            self.failures += 1
        return penalty, hard

    def _score(self, solution) -> float:
        if not self._complete(solution):
            return 1e9
        travel = self.obj.cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _initial(self) -> Dict[int, List[int]]:
        sol = self._greedy_insert(list(range(self.n)))
        if self._trivial(sol):
            order = sorted(range(self.n),
                           key=lambda c: -sum(self.data["customers"][c]["demand"].values()))
            sol = self._greedy_insert(order)
        if self._trivial(sol):
            for _ in range(3):
                perm = list(range(self.n))
                random.shuffle(perm)
                sol = self._greedy_insert(perm)
                if not self._trivial(sol):
                    break
        return sol

    def _greedy_insert(self, order: List[int]) -> Dict[int, List[int]]:
        solution: Dict[int, List[int]] = {}
        for c in order:
            best = (None, None, float("inf"))
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.obj.cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen)
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                solution[len(solution)] = [c]
            else:
                solution[vid].insert(pos, c)
        return solution

    @staticmethod
    def _trivial(solution) -> bool:
        return bool(solution) and all(len(r) == 1 for r in solution.values())

    def _two_opt(self, solution) -> Dict[int, List[int]]:
        new = {k: list(v) for k, v in solution.items()}
        for vid in list(new.keys()):
            route = new[vid]
            if len(route) < 3:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        pen, hard = self._route_penalty(nr)
                        if hard:
                            continue
                        cur = self.obj.cost({vid: route}) + (self.lambda_W + self.lambda_S) * self._route_penalty(route)[0]
                        nxt = self.obj.cost({vid: nr}) + (self.lambda_W + self.lambda_S) * pen
                        if nxt < cur:
                            route = nr
                            improved = True
                            break
                    if improved:
                        break
            new[vid] = route
        return new

    def _finalize(self, solution, iterations, wall) -> Metrics:
        m = self.obj.metrics(solution, iterations=iterations,
                             routing_time=wall,
                             lambda_W=self.lambda_W, lambda_S=self.lambda_S)
        m.feasibility_attempts = self.attempts
        m.feasibility_failures = self.failures
        m.route_penalty_cache_hits = self.cache_hits
        total = self.attempts + self.cache_hits
        m.route_penalty_cache_hit_rate = (100.0 * self.cache_hits / total) if total else 0.0
        return m


# ============================================================================
# STANDARD ALNS
# ============================================================================

class StandardALNS(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 150, "initial_temperature": 100.0,
                                 "cooling_rate": 0.99, "destruction_rate": 0.30,
                                 "max_no_improvement": 50}

    def solve(self):
        t0 = time.time()
        sol = self._initial()
        cur = self._score(sol)
        best, best_cost = {k: list(v) for k, v in sol.items()}, cur
        T = self.params["initial_temperature"]
        it = noimp = 0
        while it < self.params["max_iterations"]:
            it += 1
            d = random.choice(["random", "worst", "shaw"])
            r = random.choice(["greedy", "regret"])
            destroyed, removed = self._destroy(sol, d)
            repaired = self._repair(destroyed, removed, r)
            new = self._score(repaired)
            if new < cur or random.random() < math.exp(-(new - cur) / max(T, 1e-10)):
                sol, cur = repaired, new
                if new < best_cost:
                    best, best_cost = {k: list(v) for k, v in sol.items()}, new
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            T *= self.params["cooling_rate"]
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it, wall)

    def _destroy(self, solution, op):
        new = {k: list(v) for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]
        if op == "random":
            k = max(2, int(len(all_c) * self.params["destruction_rate"]))
            if len(all_c) <= k:
                return new, []
            removed = set(random.sample(all_c, k))
            for vid in list(new):
                new[vid] = [c for c in new[vid] if c not in removed]
                if not new[vid]:
                    del new[vid]
            return new, list(removed)
        if op == "worst":
            marginal = {}
            for vid, route in new.items():
                if len(route) <= 1:
                    continue
                for i, c in enumerate(route):
                    rw = route[:i] + route[i + 1:]
                    if rw:
                        marginal[(vid, c)] = (self.obj.cost({vid: route})
                                              - self.obj.cost({vid: rw}))
            if not marginal:
                return new, []
            k = max(2, int(len(marginal) * self.params["destruction_rate"] * 0.8))
            kept = sorted(marginal.items(), key=lambda x: x[1], reverse=True)[:k]
            removed = {key for key, _ in kept}
            for vid in list(new):
                new[vid] = [c for c in new[vid] if (vid, c) not in removed]
                if not new[vid]:
                    del new[vid]
            return new, [c for _, c in removed]
        # shaw
        if len(all_c) < 3:
            return self._destroy(solution, "random")
        seed = random.choice(all_c)
        sp = set(self.data["customers"][seed]["demand"])
        similar = []
        for c in all_c:
            if c == seed:
                continue
            ov = len(sp & set(self.data["customers"][c]["demand"]))
            if ov > 0:
                similar.append((c, ov))
        if not similar:
            return self._destroy(solution, "random")
        similar.sort(key=lambda x: x[1], reverse=True)
        k = max(2, min(len(similar), int(len(all_c) * self.params["destruction_rate"] * 0.7)))
        removed = {seed} | {c for c, _ in similar[:k]}
        for vid in list(new):
            new[vid] = [c for c in new[vid] if c not in removed]
            if not new[vid]:
                del new[vid]
        return new, list(removed)

    def _repair(self, solution, unassigned, op):
        new = {k: list(v) for k, v in solution.items()}
        remaining = list(unassigned)
        random.shuffle(remaining)
        for c in remaining[:]:
            best = (None, None, float("inf"))
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = self.obj.cost({vid: test}) + (self.lambda_W + self.lambda_S) * pen
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                new[len(new)] = [c]
            else:
                new[vid].insert(pos, c)
            remaining.remove(c)
        return new


# ============================================================================
# FDAHS
# ============================================================================

class FDAHSAlgorithm(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "population_size": 10,
                                 "crossover_rate": 0.8, "mutation_rate": 0.1,
                                 "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        pop = [self._two_opt(self._initial()) for _ in range(self.params["population_size"])]
        costs = [self._score(s) for s in pop]
        bi = int(np.argmin(costs))
        best, best_cost = {k: list(v) for k, v in pop[bi].items()}, costs[bi]
        it = noimp = 0
        while it < self.params["max_iterations"]:
            it += 1
            new_pop = []
            for _ in range(self.params["population_size"]):
                c = self._crossover(random.choice(pop), random.choice(pop))
                if random.random() < self.params["mutation_rate"]:
                    c = self._mutate(c)
                new_pop.append(self._two_opt(c))
            pop = new_pop
            costs = [self._score(s) for s in pop]
            bi = int(np.argmin(costs))
            if costs[bi] < best_cost - 1e-9:
                best, best_cost = {k: list(v) for k, v in pop[bi].items()}, costs[bi]
                noimp = 0
            else:
                noimp += 1
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it, wall)

    def _crossover(self, p1, p2):
        c1 = [c for r in p1.values() for c in r]
        c2 = [c for r in p2.values() for c in r]
        if len(c1) < 2:
            return {k: list(v) for k, v in p1.items()}
        i = random.randint(0, len(c1) - 2)
        j = random.randint(i + 1, len(c1) - 1)
        seq: List[Optional[int]] = [None] * len(c1)
        seq[i:j] = c1[i:j]
        j2 = 0
        for k in range(len(seq)):
            if seq[k] is None:
                while j2 < len(c2) and c2[j2] in seq:
                    j2 += 1
                if j2 < len(c2):
                    seq[k] = c2[j2]
                    j2 += 1
        child: Dict[int, List[int]] = {}
        for c in seq:
            if c is None:
                continue
            placed = False
            for vid, route in child.items():
                test = route + [c]
                _, hard = self._route_penalty(test)
                if not hard:
                    child[vid].append(c)
                    placed = True
                    break
            if not placed:
                child[len(child)] = [c]
        return child

    def _mutate(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        routes = list(new)
        if len(routes) < 2:
            return new
        v1, v2 = random.sample(routes, 2)
        r1, r2 = new[v1], new[v2]
        if not r1 or not r2:
            return new
        i1 = random.randint(0, len(r1) - 1)
        i2 = random.randint(0, len(r2) - 1)
        nr1 = r1[:i1] + [r2[i2]] + r1[i1 + 1:]
        nr2 = r2[:i2] + [r1[i1]] + r2[i2 + 1:]
        _, h1 = self._route_penalty(nr1)
        _, h2 = self._route_penalty(nr2)
        if not h1 and not h2:
            new[v1], new[v2] = nr1, nr2
        return new


# ============================================================================
# GCOF
# ============================================================================

class GCOFAlgorithm(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "population_size": 10,
                                 "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        pop = [self._two_opt(self._initial()) for _ in range(self.params["population_size"])]
        costs = [self._score(s) for s in pop]
        bi = int(np.argmin(costs))
        best, best_cost = {k: list(v) for k, v in pop[bi].items()}, costs[bi]
        it = noimp = 0
        while it < self.params["max_iterations"]:
            it += 1
            new_pop = []
            for sol in pop:
                g = {k: list(v) for k, v in sol.items()}
                if random.random() < 0.3:
                    g = self._cgpa(g)
                new_pop.append(self._two_opt(g))
            pop = new_pop
            costs = [self._score(s) for s in pop]
            bi = int(np.argmin(costs))
            if costs[bi] < best_cost - 1e-9:
                best, best_cost = {k: list(v) for k, v in pop[bi].items()}, costs[bi]
                noimp = 0
            else:
                noimp += 1
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it, wall)

    def _cgpa(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    test = route[:i] + route[j:j + 1] + route[i:j] + route[j + 1:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    cur = self.obj.cost({vid: route}) + (self.lambda_W + self.lambda_S) * self._route_penalty(route)[0]
                    nxt = self.obj.cost({vid: test}) + (self.lambda_W + self.lambda_S) * pen
                    if nxt < cur:
                        new[vid] = test
                        break
        return new


# ============================================================================
# HSA
# ============================================================================

class HSAAlgorithm(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "initial_temperature": 40.0,
                                 "cooling_rate": 0.98, "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        sol = self._two_opt(self._initial())
        cur = self._score(sol)
        best, best_cost = {k: list(v) for k, v in sol.items()}, cur
        T = self.params["initial_temperature"]
        it = noimp = 0
        while T > 0.001 and it < self.params["max_iterations"]:
            it += 1
            nb = self._neighbor(sol)
            nc = self._score(nb)
            if nc < cur or random.random() < math.exp(-(nc - cur) / max(T, 1e-10)):
                sol, cur = nb, nc
                if nc < best_cost:
                    best, best_cost = {k: list(v) for k, v in sol.items()}, nc
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            T *= self.params["cooling_rate"]
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it, wall)

    def _neighbor(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        routes = list(new)
        if len(routes) < 2:
            return new
        v1, v2 = random.sample(routes, 2)
        r1, r2 = new[v1], new[v2]
        if not r1 or not r2:
            return new
        i1 = random.randint(0, len(r1) - 1)
        i2 = random.randint(0, len(r2) - 1)
        nr1 = r1[:i1] + [r2[i2]] + r1[i1 + 1:]
        nr2 = r2[:i2] + [r1[i1]] + r2[i2 + 1:]
        _, h1 = self._route_penalty(nr1)
        _, h2 = self._route_penalty(nr2)
        if not h1 and not h2:
            new[v1], new[v2] = nr1, nr2
        return new


# ============================================================================
# ACO-2opt
# ============================================================================

class ACO2optAlgorithm(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "num_ants": 10,
                                 "alpha": 1.0, "beta": 2.0, "rho": 0.9,
                                 "max_no_improvement": 25}
        self.pheromone: Optional[np.ndarray] = None

    def solve(self):
        t0 = time.time()
        n = self.n
        self.pheromone = np.full((n + 1, n + 1), 1.0 / (n + 1))
        best = self._two_opt(self._initial())
        best_cost = self._score(best)
        it = noimp = 0
        for it in range(self.params["max_iterations"]):
            sols, costs = [], []
            for _ in range(self.params["num_ants"]):
                s = self._two_opt(self._ant())
                c = self._score(s)
                sols.append(s)
                costs.append(c)
                if c < best_cost:
                    best, best_cost = {k: list(v) for k, v in s.items()}, c
                    noimp = 0
                else:
                    noimp += 1
            self._update_pheromone(sols, costs)
            self.pheromone *= (1 - self.params["rho"])
            self.pheromone += 1e-9
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it + 1, wall)

    def _ant(self):
        order = list(range(self.n))
        random.shuffle(order)
        sol: Dict[int, List[int]] = {}
        for c in order:
            best = None
            for vid, route in sol.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    _, hard = self._route_penalty(test)
                    if hard:
                        continue
                    prev = 0 if pos == 0 else route[pos - 1] + 1
                    nxt = 0 if pos == len(route) else route[pos] + 1
                    h = 1.0 / (self.dm[prev][c + 1] + self.dm[c + 1][nxt] + 1e-6)
                    pher = 0.5 * (self.pheromone[prev][c + 1] + self.pheromone[c + 1][nxt])
                    sc = (pher ** self.params["alpha"]) * (h ** self.params["beta"])
                    if best is None or sc > best[0]:
                        best = (sc, vid, pos)
            if best is None:
                sol[len(sol)] = [c]
            else:
                _, vid, pos = best
                sol[vid].insert(pos, c)
        return sol

    def _update_pheromone(self, sols, costs):
        for s, c in zip(sols, costs):
            if c <= 0 or not np.isfinite(c):
                continue
            d = 1.0 / c
            for route in s.values():
                if not route:
                    continue
                nodes = [0] + [x + 1 for x in route] + [0]
                for i in range(len(nodes) - 1):
                    self.pheromone[nodes[i]][nodes[i + 1]] += d
                    self.pheromone[nodes[i + 1]][nodes[i]] += d


# ============================================================================
# SA-ALNS
# ============================================================================

class SAALNS(BaseAlgorithm):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, stability_weight=stability_weight)
        self.params = params or {
            "max_iterations": 150, "initial_temperature": 100.0,
            "cooling_rate": 0.99, "destruction_rate": 0.30,
            "learning_rate": 0.15, "stability_weight": stability_weight,
            "adaptive_frequency": 10, "max_no_improvement": 50,
        }
        self.saca.stability_weight = self.params["stability_weight"]
        self.destroy_weights = {"random": 1.0, "worst": 1.2, "compatibility": 1.0,
                                "imbalance": 1.3, "shaw": 1.1}
        self.repair_weights = {"greedy": 1.0, "regret": 1.3,
                               "stability_aware": 1.5, "balance": 1.1}
        self.op_stats = {
            "destroy": {op: {"selected": 0, "improved": 0, "accepted": 0}
                        for op in self.destroy_weights},
            "repair": {op: {"selected": 0, "improved": 0, "accepted": 0}
                       for op in self.repair_weights},
        }

    def solve(self):
        t0 = time.time()
        sol = self._enhanced_initial()
        cur = self._score(sol)
        best, best_cost = {k: list(v) for k, v in sol.items()}, cur
        T = self.params["initial_temperature"]
        it = noimp = 0
        dperf: Dict[str, List[float]] = defaultdict(list)
        rperf: Dict[str, List[float]] = defaultdict(list)
        while it < self.params["max_iterations"]:
            it += 1
            d = self._pick(self.destroy_weights)
            r = self._pick(self.repair_weights)
            self.op_stats["destroy"][d]["selected"] += 1
            self.op_stats["repair"][r]["selected"] += 1
            destroyed, removed = self._destroy(sol, d)
            repaired = self._repair(destroyed, removed, r)
            new_cost = self._score(repaired)
            if cur > 0 and np.isfinite(cur) and np.isfinite(new_cost):
                imp = (cur - new_cost) / cur
                dperf[d].append(imp)
                rperf[r].append(imp)
            accepted = (new_cost < cur or
                        random.random() < math.exp(-(new_cost - cur) / max(T, 1e-10)))
            if accepted:
                self.op_stats["destroy"][d]["accepted"] += 1
                self.op_stats["repair"][r]["accepted"] += 1
                if new_cost < best_cost:
                    self.op_stats["destroy"][d]["improved"] += 1
                    self.op_stats["repair"][r]["improved"] += 1
                sol, cur = repaired, new_cost
                if new_cost < best_cost:
                    best, best_cost = {k: list(v) for k, v in sol.items()}, new_cost
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            if it % self.params["adaptive_frequency"] == 0:
                self._update_weights(dperf, rperf)
                dperf.clear()
                rperf.clear()
            T *= self.params["cooling_rate"]
            self.history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall = time.time() - t0
        return best, best_cost, self._finalize(best, it, wall)

    def _enhanced_initial(self):
        order = sorted(range(self.n),
                       key=lambda c: -sum(self.data["customers"][c]["demand"].values()))
        sol: Dict[int, List[int]] = {}
        for c in order:
            best = (None, None, float("inf"))
            for vid, route in sol.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.obj.cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen
                             + 0.5 * self._slosh(test))
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                sol[len(sol)] = [c]
            else:
                sol[vid].insert(pos, c)
        if self._trivial(sol):
            return self._initial()
        return sol

    def _slosh(self, route) -> float:
        demands = {c: self.data["customers"][c]["demand"] for c in route}
        travel = self.obj.cost({0: route})
        pen, _, _, hard = self.saca.verify(route, demands, transport_cost=travel,
                                           skip_sequence_check=True)
        return 0.0 if hard else pen

    @property
    def ldd(self):
        return self.saca.ldd

    def _pick(self, weights) -> str:
        if random.random() < 0.15:
            return random.choice(list(weights))
        total = sum(weights.values()) or 1.0
        r = random.random() * total
        acc = 0.0
        for op, w in weights.items():
            acc += w
            if r <= acc:
                return op
        return list(weights)[-1]

    def _destroy(self, solution, op):
        if op == "random":
            return self._destroy_random(solution)
        if op == "worst":
            return self._destroy_worst(solution)
        if op == "compatibility":
            return self._destroy_compat(solution)
        if op == "imbalance":
            return self._destroy_imbalance(solution)
        return self._destroy_shaw(solution)

    def _destroy_random(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]
        k = max(2, int(len(all_c) * self.params["destruction_rate"]))
        if len(all_c) <= k:
            return new, []
        removed = set(random.sample(all_c, k))
        for vid in list(new):
            new[vid] = [c for c in new[vid] if c not in removed]
            if not new[vid]:
                del new[vid]
        return new, list(removed)

    def _destroy_worst(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        marginal = {}
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i, c in enumerate(route):
                rw = route[:i] + route[i + 1:]
                if rw:
                    marginal[(vid, c)] = (self.obj.cost({vid: route})
                                          - self.obj.cost({vid: rw}))
        if not marginal:
            return new, []
        k = max(2, int(len(marginal) * self.params["destruction_rate"] * 0.8))
        removed = {x[0] for x in sorted(marginal.items(), key=lambda x: x[1], reverse=True)[:k]}
        for vid in list(new):
            new[vid] = [c for c in new[vid] if (vid, c) not in removed]
            if not new[vid]:
                del new[vid]
        return new, [c for _, c in removed]

    def _destroy_shaw(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]
        if len(all_c) < 3:
            return self._destroy_random(solution)
        seed = random.choice(all_c)
        sp = set(self.data["customers"][seed]["demand"])
        similar = []
        for c in all_c:
            if c == seed:
                continue
            ov = len(sp & set(self.data["customers"][c]["demand"]))
            if ov > 0:
                similar.append((c, ov))
        if not similar:
            return self._destroy_random(solution)
        similar.sort(key=lambda x: x[1], reverse=True)
        k = max(2, min(len(similar), int(len(all_c) * self.params["destruction_rate"] * 0.7)))
        removed = {seed} | {c for c, _ in similar[:k]}
        for vid in list(new):
            new[vid] = [c for c in new[vid] if c not in removed]
            if not new[vid]:
                del new[vid]
        return new, list(removed)

    def _destroy_compat(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        inc: Dict[int, List[int]] = defaultdict(list)
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i, c1 in enumerate(route):
                for c2 in route[i + 1:]:
                    p1 = set(self.data["customers"][c1]["demand"])
                    p2 = set(self.data["customers"][c2]["demand"])
                    if p1 != p2:
                        inc[vid].extend([c1, c2])
        if not inc:
            return new, []
        target = random.choice(list(inc))
        unique = list(set(inc[target]))
        k = max(2, min(len(unique), int(len(unique) * 0.5)))
        removed = set(random.sample(unique, k))
        new[target] = [c for c in new[target] if c not in removed]
        if not new[target]:
            del new[target]
        return new, list(removed)

    def _destroy_imbalance(self, solution):
        new = {k: list(v) for k, v in solution.items()}
        unstable = []
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            pen, hard = self._route_penalty(route)
            if hard or pen > 1e-6:
                unstable.append(vid)
        if not unstable:
            return self._destroy_random(solution)
        target = random.choice(unstable)
        route = new[target]
        for i, c in enumerate(route):
            test = route[:i] + route[i + 1:]
            pen, hard = self._route_penalty(test)
            if not hard:
                new[target] = [x for x in route if x != c]
                if not new[target]:
                    del new[target]
                return new, [c]
        return self._destroy_random(solution)

    def _repair(self, solution, unassigned, op):
        if op == "greedy":
            return self._repair_greedy(solution, unassigned)
        if op == "regret":
            return self._repair_regret(solution, unassigned)
        if op == "stability_aware":
            return self._repair_stab(solution, unassigned)
        return self._repair_balance(solution, unassigned)

    def _repair_greedy(self, solution, unassigned):
        new = {k: list(v) for k, v in solution.items()}
        rem = list(unassigned)
        random.shuffle(rem)
        for c in rem[:]:
            best = (None, None, float("inf"))
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = self.obj.cost({vid: test}) + (self.lambda_W + self.lambda_S) * pen
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                new[len(new)] = [c]
            else:
                new[vid].insert(pos, c)
            rem.remove(c)
        return new

    def _repair_regret(self, solution, unassigned):
        new = {k: list(v) for k, v in solution.items()}
        rem = list(unassigned)
        while rem:
            regret_rows = []
            for c in rem:
                options = []
                for vid, route in new.items():
                    for pos in range(len(route) + 1):
                        test = route[:pos] + [c] + route[pos:]
                        pen, hard = self._route_penalty(test)
                        if not hard:
                            options.append(self.obj.cost({vid: test})
                                           + (self.lambda_W + self.lambda_S) * pen)
                if len(options) >= 2:
                    options.sort()
                    regret_rows.append((options[1] - options[0], c))
                elif options:
                    regret_rows.append((options[0], c))
                else:
                    regret_rows.append((float("inf"), c))
            if not regret_rows:
                break
            regret_rows.sort(key=lambda x: x[0], reverse=True)
            if regret_rows[0][0] == float("inf"):
                c = rem.pop(0)
                new[len(new)] = [c]
                continue
            c = regret_rows[0][1]
            best = (None, None, float("inf"))
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = self.obj.cost({vid: test}) + (self.lambda_W + self.lambda_S) * pen
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                new[len(new)] = [c]
            else:
                new[vid].insert(pos, c)
            rem.remove(c)
        return new

    def _repair_stab(self, solution, unassigned):
        new = {k: list(v) for k, v in solution.items()}
        rem = list(unassigned)
        for c in rem[:]:
            best = (None, None, float("inf"))
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.obj.cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen
                             + self.params["stability_weight"] * self._slosh(test))
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                new[len(new)] = [c]
            else:
                new[vid].insert(pos, c)
            rem.remove(c)
        return new

    def _repair_balance(self, solution, unassigned):
        new = {k: list(v) for k, v in solution.items()}
        rem = list(unassigned)
        random.shuffle(rem)
        for c in rem[:]:
            min_load = float("inf")
            best = (None, None)
            for vid, route in new.items():
                load = sum(sum(self.data["customers"][x]["demand"].values()) for x in route)
                if load < min_load:
                    for pos in range(len(route) + 1):
                        test = route[:pos] + [c] + route[pos:]
                        _, hard = self._route_penalty(test)
                        if not hard:
                            min_load = load
                            best = (vid, pos)
                            break
            vid, pos = best
            if vid is None:
                new[len(new)] = [c]
            else:
                new[vid].insert(pos, c)
            rem.remove(c)
        return new

    def _update_weights(self, dperf, rperf):
        lr = self.params["learning_rate"]
        for op, s in dperf.items():
            if s:
                self.destroy_weights[op] = (1 - lr) * self.destroy_weights[op] + lr * max(0.1, float(np.mean(s)) * 2)
        for op, s in rperf.items():
            if s:
                self.repair_weights[op] = (1 - lr) * self.repair_weights[op] + lr * max(0.1, float(np.mean(s)) * 2)
        td = sum(self.destroy_weights.values()) or 1.0
        tr = sum(self.repair_weights.values()) or 1.0
        for op in self.destroy_weights:
            self.destroy_weights[op] /= td
        for op in self.repair_weights:
            self.repair_weights[op] /= tr


class SAALNSNoStability(SAALNS):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, params, stability_weight=0.0)
        self.stability_weight = 0.0
        self.saca.stability_weight = 0.0
        self.params["stability_weight"] = 0.0

    def _slosh(self, route):
        return 0.0


class SAALNSNoAdaptive(SAALNS):
    def _pick(self, weights):
        return random.choice(list(weights))

    def _update_weights(self, dperf, rperf):
        pass


class SAALNSSequential(SAALNS):
    def solve(self):
        t0 = time.time()
        routing = self._solve_routing_only()
        final = self._solve_loading(routing)
        wall = time.time() - t0
        m = self._finalize(final, len(self.history), wall)
        return final, m.penalised_cost, m

    def _total(self, customer) -> float:
        return sum(self.data["customers"][customer]["demand"].values())

    def _fits(self, route) -> bool:
        return sum(self._total(c) for c in route) <= self.vehicle.max_volume + 1e-6

    def _routing_cost(self, solution) -> float:
        return self.obj.cost(solution)

    def _routing_only_insert(self, order):
        sol: Dict[int, List[int]] = {}
        for c in order:
            best = (None, None, float("inf"))
            for vid, route in sol.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    if not self._fits(test):
                        continue
                    score = self._routing_cost({vid: test})
                    if score < best[2]:
                        best = (vid, pos, score)
            vid, pos, _ = best
            if vid is None:
                sol[len(sol)] = [c]
            else:
                sol[vid].insert(pos, c)
        return sol

    def _solve_routing_only(self):
        order = sorted(range(self.n), key=lambda c: -self._total(c))
        sol = self._routing_only_insert(order)
        if self._trivial(sol):
            shuffled = list(range(self.n))
            random.shuffle(shuffled)
            sol = self._routing_only_insert(shuffled)
        for vid in list(sol):
            route = sol[vid]
            if len(route) < 3:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if not self._fits(nr):
                            continue
                        if self._routing_cost({vid: nr}) < self._routing_cost({vid: route}):
                            sol[vid] = nr
                            route = nr
                            improved = True
                            break
                    if improved:
                        break
        self.history.append(self._routing_cost(sol))
        return sol

    def _solve_loading(self, routing):
        final: Dict[int, List[int]] = {}
        for _, route in routing.items():
            if not route:
                continue
            demands = {c: self.data["customers"][c]["demand"] for c in route}
            travel = self.obj.cost({0: route})
            pen, _, _, hard = self.saca.verify(route, demands, transport_cost=travel,
                                              skip_sequence_check=True)
            if not hard and pen < 1e-6:
                final[len(final)] = route
                continue
            split = self._best_split(route)
            if split:
                final[len(final)] = split[0]
                final[len(final)] = split[1]
            else:
                for c in route:
                    final[len(final)] = [c]
        return final

    def _best_split(self, route):
        if len(route) <= 1:
            return None
        best, best_cost = None, float("inf")
        for i in range(1, len(route)):
            r1, r2 = route[:i], route[i:]
            d1 = {c: self.data["customers"][c]["demand"] for c in r1}
            d2 = {c: self.data["customers"][c]["demand"] for c in r2}
            t1 = self.obj.cost({0: r1})
            t2 = self.obj.cost({1: r2})
            p1, _, _, h1 = self.saca.verify(r1, d1, transport_cost=t1, skip_sequence_check=True)
            p2, _, _, h2 = self.saca.verify(r2, d2, transport_cost=t2, skip_sequence_check=True)
            if h1 or h2:
                continue
            cand = t1 + t2 + (self.lambda_W + self.lambda_S) * (p1 + p2)
            if cand < best_cost:
                best_cost, best = cand, [r1, r2]
        return best


class SAALNSNoCache(SAALNS):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, params, stability_weight=stability_weight)
        new_saca = SACA(vehicle, data, stability_weight=self.stability_weight,
                        use_cache=False)
        self.saca = new_saca
        self.obj.saca = new_saca

    def _route_penalty(self, route):
        if not route:
            return 0.0, False
        self.attempts += 1
        demands = {c: self.data["customers"][c]["demand"] for c in route}
        if not self.saca._sequence_feasible(route, demands):
            self.failures += 1
            return 1e6, True
        travel = self.obj.cost({0: route})
        penalty, _, _, hard = self.saca.verify(route, demands, transport_cost=travel,
                                              skip_sequence_check=True)
        if hard or penalty > 1e-6:
            self.failures += 1
        return penalty, hard


class SAALNSNoImbalance(SAALNS):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, params, stability_weight=stability_weight)
        self.destroy_weights = {"random": 1.0, "worst": 1.2,
                                "compatibility": 1.0, "shaw": 1.1}
        self.op_stats = {
            "destroy": {op: {"selected": 0, "improved": 0, "accepted": 0}
                        for op in self.destroy_weights},
            "repair": {op: {"selected": 0, "improved": 0, "accepted": 0}
                       for op in self.repair_weights},
        }


class SAALNSSingleProduct(SAALNS):
    def __init__(self, data, vehicle, params=None, stability_weight=0.35):
        super().__init__(data, vehicle, params, stability_weight=stability_weight)
        new_saca = SACASingleProduct(vehicle, data,
                                     stability_weight=self.stability_weight)
        self.saca = new_saca
        self.obj.saca = new_saca


# Backward-compatible aliases
IALNS = SAALNS
IALNS_NoStability = SAALNSNoStability
IALNS_NoAdaptive = SAALNSNoAdaptive
IALNS_Sequential = SAALNSSequential
IALNS_NoCache = SAALNSNoCache
IALNS_NoImbalance = SAALNSNoImbalance
IALNS_SingleProduct = SAALNSSingleProduct


# ============================================================================
# COMPARISON RUNNERS
# ============================================================================

def _finite_mean(series) -> float:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else float("nan")


def _wilcoxon_safe(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return float("nan")
    try:
        return float(wilcoxon(a, b)[1])
    except Exception:
        return float("nan")


def _mannwhitney_safe(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        return float(mannwhitneyu(a, b)[1])
    except Exception:
        return float("nan")


def _list_instances(directory: str) -> List[str]:
    files = [os.path.join(root, f) for root, _, fs in os.walk(directory) for f in fs if f.endswith(".json")]
    files.sort(key=lambda p: (len(json.load(open(p))["customers"]), os.path.basename(p)))
    return files


def _run_one(algo_cls, data, vehicle, name, n_expected, seeds=1):
    costs, times, metrics, infeasible = [], [], [], 0
    last_algo = None
    for run in range(seeds):
        random.seed(stable_seed(name, run, data["name"]))
        np.random.seed(stable_seed(name, run, data["name"]))
        try:
            algo = algo_cls(data, vehicle)
            last_algo = algo
            t0 = time.time()
            sol, cost, m = algo.solve()
            elapsed = time.time() - t0
            if sol:
                counts = sum(len(r) for r in sol.values() if r)
                if counts != n_expected:
                    print(f" [ALERT {name}: {counts}/{n_expected}]", end="")
                if np.isfinite(m.penalised_cost):
                    costs.append(m.penalised_cost)
                    times.append(elapsed)
                    metrics.append(m)
                else:
                    infeasible += 1
                    times.append(elapsed)
                    metrics.append(m)
        except Exception as e:
            print(f" [err {e}]", end="")
    return costs, times, metrics, infeasible, last_algo


def _avg_metrics(metrics: List[Metrics]) -> Metrics:
    if not metrics:
        return Metrics()
    attrs = [f for f in Metrics.__dataclass_fields__ if f != "op_stats"]
    avg = Metrics()
    for m in metrics:
        for a in attrs:
            setattr(avg, a, getattr(avg, a) + getattr(m, a, 0.0))
    n = len(metrics)
    for a in attrs:
        setattr(avg, a, getattr(avg, a) / n)
    for a in ("cg_compliance_rate", "fleet_utilization", "volume_utilization",
              "ldd_violation_rate", "stability_margin", "minimum_load_ratio",
              "cache_hit_rate", "route_penalty_cache_hit_rate"):
        setattr(avg, a, max(0.0, min(100.0, getattr(avg, a))))
    return avg


def run_sota(instances: List[str], out_dir: str, seeds: int,
             algorithms: Dict[str, type]) -> pd.DataFrame:
    rows = []
    for i, fp in enumerate(instances):
        inst = json.load(open(fp))
        vehicle = load_vehicle(inst)
        data = build_data(inst)
        n_expected = len(inst["customers"])
        print(f"\n[{i + 1}/{len(instances)}] {os.path.basename(fp)}")
        for name, cls in algorithms.items():
            print(f"    {name}...", end="", flush=True)
            costs, times, metrics, infeasible, last_algo = _run_one(
                cls, data, vehicle, name, n_expected, seeds)
            if not metrics:
                print(" failed", flush=True)
                continue
            avg = _avg_metrics(metrics)
            finite = [c for c in costs if np.isfinite(c)]
            rows.append({
                "instance": inst["name"], "algorithm": name,
                "n_customers": n_expected,
                "n_compartments": inst.get("n_compartments", len(inst["vehicle"]["compartments"])),
                "layer": inst.get("layer", "Unknown"),
                "n_products": inst.get("n_products", 4),
                "avg_cost": float(np.mean(finite)) if finite else float("inf"),
                "std_cost": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "n_infeasible_runs": infeasible,
                "infeasibility_rate_pct": 100.0 * infeasible / max(len(metrics), 1),
                "avg_total_time": float(np.mean(times)) if times else 0.0,
                "avg_milp_calls": avg.milp_calls,
                "avg_milp_time": avg.milp_time,
                "avg_cache_hit_rate": avg.cache_hit_rate,
                "avg_route_penalty_cache_hit_rate": avg.route_penalty_cache_hit_rate,
                "avg_ldd_violation_rate": avg.ldd_violation_rate,
                "avg_stability": avg.stability_margin,
                "avg_slosh_risk": avg.slosh_risk_score,
                "avg_cg_compliance_rate": avg.cg_compliance_rate,
                "avg_cg_deviation": avg.cg_deviation_from_optimum,
                "avg_volume_utilization": avg.volume_utilization,
                "avg_fleet_size": avg.fleet_size,
                "avg_feasibility_failures": avg.feasibility_failures,
                "avg_feasibility_attempts": avg.feasibility_attempts,
                "feasibility_failure_rate": 100.0 * avg.feasibility_failures / max(1, avg.feasibility_attempts),
                "avg_penalty": avg.avg_penalty,
                "avg_n_total_routes": float(avg.n_total_routes),
                "avg_n_verified_routes": float(avg.n_verified_routes),
            })
            c = rows[-1]["avg_cost"]
            print(f" done ({c:.2f})" if np.isfinite(c) else " done (inf)", flush=True)

            if name == "SA-ALNS" and last_algo is not None and hasattr(last_algo, "op_stats"):
                _write_op_stats(last_algo.op_stats, out_dir)
    return pd.DataFrame(rows)


def _write_op_stats(op_stats: Dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "operator_stats.json")
    with open(path, "w") as f:
        json.dump(op_stats, f, indent=2)
    print(f"\nOperator statistics written to {path}")


def run_statistics(sota: pd.DataFrame, out_dir: str) -> None:
    sa = sota[sota["algorithm"] == "SA-ALNS"]
    if sa.empty:
        return
    scost = sa["avg_cost"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(scost) < 5:
        return
    rows = []
    for algo in ("SA-ALNS_CostOnly", "Standard_ALNS", "FDAHS", "GCOF", "HSA", "ACO_2opt"):
        sub = sota[sota["algorithm"] == algo]
        if sub.empty:
            continue
        c = sub["avg_cost"].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(c) < 5:
            continue
        m1, m2 = float(np.mean(scost)), float(np.mean(c))
        s1 = float(np.std(scost, ddof=1)) if len(scost) > 1 else 0.0
        s2 = float(np.std(c, ddof=1)) if len(c) > 1 else 0.0
        pooled = math.sqrt(((len(scost) - 1) * s1 ** 2 + (len(c) - 1) * s2 ** 2)
                           / max(len(scost) + len(c) - 2, 1))
        rows.append({
            "Algorithm": SOTA_DISPLAY_NAMES.get(algo, algo),
            "Comp_mean": m2,
            "Improvement_pct": (m2 - m1) / m2 * 100 if m2 > 0 else 0.0,
            "Wilcoxon_p": _wilcoxon_safe(scost, c),
            "MannWhitney_p": _mannwhitney_safe(scost, c),
            "Cohen_d": (m1 - m2) / pooled if pooled > 0 else 0.0,
        })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "part4_statistics",
                                                "statistical_tests_results.csv"), index=False)


def paired_ablation(out_dir: str) -> None:
    path = os.path.join(out_dir, "ablation_study_results.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df["avg_cost"] = df["avg_cost"].replace([np.inf, -np.inf], np.nan)
    pivot = df.pivot_table(index="instance", columns="algorithm",
                           values="avg_cost", aggfunc="mean")
    if "SA-ALNS" not in pivot.columns:
        return
    rows = []
    for algo in pivot.columns:
        if algo == "SA-ALNS":
            continue
        pair = pivot[["SA-ALNS", algo]].dropna()
        if len(pair) < 5:
            continue
        diff = pair["SA-ALNS"] - pair[algo]
        rows.append({
            "Variant": ABLATION_DISPLAY_NAMES.get(algo, algo),
            "n_pairs": len(pair),
            "SAALNS_mean": float(pair["SA-ALNS"].mean()),
            "Variant_mean": float(pair[algo].mean()),
            "Mean_delta": float(diff.mean()),
            "Median_delta": float(diff.median()),
            "Wilcoxon_p": _wilcoxon_safe(pair["SA-ALNS"].values, pair[algo].values),
            "SAALNS_wins": int((diff < 0).sum()),
            "Variant_wins": int((diff > 0).sum()),
            "Ties": int((np.abs(diff) < 1e-9).sum()),
        })
    if rows:
        pd.DataFrame(rows).sort_values("Wilcoxon_p", na_position="last").to_csv(
            os.path.join(out_dir, "part4_statistics", "ablation_paired_tests.csv"), index=False)


def run_multicomp(sota: pd.DataFrame, base_dir: str, out_dir: str) -> pd.DataFrame:
    sa_rows = sota[sota["algorithm"] == "SA-ALNS"]
    if sa_rows.empty:
        return pd.DataFrame()
    rows = []
    for _, r in sa_rows.iterrows():
        name = r["instance"]
        fp = next((os.path.join(root, f"{name}.json") for root, _, fs in os.walk(base_dir)
                   if f"{name}.json" in fs), None)
        if fp is None:
            continue
        inst = json.load(open(fp))
        data = build_data(inst)
        seed = stable_seed("multicomp", name)

        def run(cls, veh):
            random.seed(seed)
            np.random.seed(seed)
            algo = cls(data, veh)
            sol, _, m = algo.solve()
            return m

        m_a = run(SAALNS, load_vehicle(inst, True))
        m_b = run(SAALNSSingleProduct, load_vehicle(inst, True))
        m_c = run(SAALNS, load_vehicle(inst, False))

        def pct(a, b):
            if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
                return float("nan")
            return (a - b) / b * 100.0

        rows.append({
            "instance": name, "n_customers": r["n_customers"],
            "layer": r.get("layer", "Unknown"),
            "n_compartments": r["n_compartments"],
            "A_full_cost": m_a.penalised_cost, "A_full_fleet": m_a.fleet_size,
            "A_full_util": m_a.volume_utilization, "A_full_cg": m_a.cg_compliance_rate,
            "A_full_ldd": m_a.ldd_violation_rate, "A_full_slosh": m_a.slosh_risk_score,
            "A_full_feasible": np.isfinite(m_a.penalised_cost),
            "B_nomix_cost": m_b.penalised_cost, "B_nomix_fleet": m_b.fleet_size,
            "B_nomix_util": m_b.volume_utilization, "B_nomix_cg": m_b.cg_compliance_rate,
            "B_nomix_ldd": m_b.ldd_violation_rate, "B_nomix_slosh": m_b.slosh_risk_score,
            "B_nomix_feasible": np.isfinite(m_b.penalised_cost),
            "C_forced_cost": m_c.penalised_cost, "C_forced_fleet": m_c.fleet_size,
            "C_forced_feasible": np.isfinite(m_c.penalised_cost),
            "cost_improvement_AB": pct(m_a.penalised_cost, m_b.penalised_cost),
            "fleet_reduction_AB": pct(m_a.fleet_size, m_b.fleet_size),
            "util_improvement_AB": pct(m_a.volume_utilization, m_b.volume_utilization),
            "cost_improvement_AC": pct(m_a.penalised_cost, m_c.penalised_cost),
            "fleet_reduction_AC": pct(m_a.fleet_size, m_c.fleet_size),
        })
        print(f"    {name}: A={m_a.penalised_cost:.1f} B={m_b.penalised_cost:.1f} C={m_c.penalised_cost:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "part4_statistics", "multicomp_benefits.csv"), index=False)
    return df


def paired_multicomp(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    rows = []
    for label, ca, cb in (("Cost", "A_full_cost", "B_nomix_cost"),
                          ("Fleet", "A_full_fleet", "B_nomix_fleet")):
        if ca not in df.columns or cb not in df.columns:
            continue
        pair = df[[ca, cb]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 5:
            continue
        diff = pair[ca] - pair[cb]
        rows.append({
            "Metric": label, "n_pairs": len(pair),
            "A_mean": float(pair[ca].mean()), "B_mean": float(pair[cb].mean()),
            "Mean_delta": float(diff.mean()), "Median_delta": float(diff.median()),
            "Wilcoxon_p": _wilcoxon_safe(pair[ca].values, pair[cb].values),
            "A_wins": int((diff < 0).sum()),
            "B_wins": int((diff > 0).sum()),
            "Ties": int((np.abs(diff) < 1e-9).sum()),
        })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "part4_statistics",
                                                "multicomp_paired_test_AB.csv"), index=False)


def run_sensitivity(base_dir: str, out_dir: str, max_instances: int = 5,
                    num_runs: int = 2) -> None:
    files = _list_instances(base_dir)[:max_instances]
    results = []
    grid = [(lw, ls) for lw in (0.50, 1.00, 1.50) for ls in (0.25, 0.50, 0.75)]
    for idx, (lw, ls) in enumerate(grid, start=1):
        print(f"  [{idx}/{len(grid)}] W={lw} S={ls}")
        costs, stabs, viols, fleets, cgs, cpus, convs = ([] for _ in range(7))
        for fp in files:
            inst = json.load(open(fp))
            veh = load_vehicle(inst)
            data = build_data(inst)
            for run in range(num_runs):
                random.seed(stable_seed("sens", lw, ls, run, fp))
                np.random.seed(stable_seed("sens", lw, ls, run, fp))
                params = {"max_iterations": 100, "initial_temperature": 100.0,
                          "cooling_rate": 0.99, "destruction_rate": 0.30,
                          "learning_rate": 0.15, "stability_weight": ls,
                          "adaptive_frequency": 10, "max_no_improvement": 40}
                try:
                    algo = SAALNS(data, veh, params, stability_weight=ls)
                    algo.lambda_W = lw
                    algo.lambda_S = ls
                    t0 = time.time()
                    sol, _, m = algo.solve()
                    elapsed = time.time() - t0
                    if sol and np.isfinite(m.penalised_cost):
                        costs.append(m.penalised_cost)
                        stabs.append(m.stability_margin)
                        viols.append(m.ldd_violation_rate)
                        fleets.append(m.fleet_size)
                        cgs.append(m.cg_compliance_rate)
                        cpus.append(elapsed)
                        convs.append(len(algo.history))
                except Exception:
                    continue
        if costs:
            results.append(SensitivityResult(
                lambda_W=lw, lambda_S=ls,
                avg_cost=float(np.mean(costs)),
                avg_stability=float(np.mean(stabs)),
                avg_ldd_violation=float(np.mean(viols)),
                avg_fleet_size=float(np.mean(fleets)),
                avg_cg_compliance=float(np.mean(cgs)),
                avg_cpu_time=float(np.mean(cpus)),
                std_cost=float(np.std(costs)),
                std_stability=float(np.std(stabs)),
                convergence_iterations=int(np.mean(convs)) if convs else 0,
                parameter_combination_id=f"W{lw}_S{ls}"))
    pd.DataFrame([r.__dict__ for r in results]).to_csv(
        os.path.join(out_dir, "part3_sensitivity", "sensitivity_analysis_results.csv"), index=False)


def run_scalability(base_dir: str, out_dir: str,
                    algorithms: Dict[str, type]) -> pd.DataFrame:
    files = _list_instances(base_dir)
    rows = []
    for i, fp in enumerate(files):
        inst = json.load(open(fp))
        veh = load_vehicle(inst)
        data = build_data(inst)
        for name, cls in algorithms.items():
            random.seed(stable_seed("scal", name, inst["name"]))
            np.random.seed(stable_seed("scal", name, inst["name"]))
            try:
                algo = cls(data, veh)
                t0 = time.time()
                sol, _, m = algo.solve()
                elapsed = time.time() - t0
                rows.append({"instance": inst["name"], "algorithm": name,
                             "n_customers": len(inst["customers"]),
                             "avg_cost": m.penalised_cost,
                             "avg_total_time": elapsed,
                             "avg_fleet_size": m.fleet_size,
                             "avg_cg_compliance_rate": m.cg_compliance_rate,
                             "avg_ldd_violation_rate": m.ldd_violation_rate,
                             "avg_stability": m.stability_margin})
            except Exception:
                continue
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(files)}] processed")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "supplementary", "sota_scalability.csv"), index=False)
    return df


# ============================================================================
# FIGURES
# ============================================================================

def fig_sota_cost(df: pd.DataFrame, out: str, fmt: str = "pdf") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_cost"].agg(["mean", "std"])
             .reset_index().dropna())
        if g.empty:
            continue
        ax.errorbar(g["n_customers"], g["mean"], yerr=g["std"].fillna(0),
                    marker=MARKERS.get(algo, "o"), color=COLORS_SOTA.get(algo, "#333"),
                    label=SOTA_DISPLAY_NAMES.get(algo, algo),
                    markersize=8, capsize=3, linewidth=3 if algo == "SA-ALNS" else 2)
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Average penalised cost (\\$)")
    ax.set_title("SOTA comparison: penalised cost vs instance size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(out, "part1_sota", f"Figure_1_Cost.{fmt}"),
                 save_format=fmt)
    plt.close(fig)


def fig_sota_timing(df: pd.DataFrame, out: str, fmt: str = "pdf") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_total_time"].mean()
             .reset_index().dropna())
        if g.empty:
            continue
        ax.plot(g["n_customers"], g["avg_total_time"], marker="o",
                color=COLORS_SOTA.get(algo, "#333"),
                label=SOTA_DISPLAY_NAMES.get(algo, algo), linewidth=2)
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_yscale("log")
    ax.set_title("Wall-clock time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(out, "part1_sota", f"Figure_3_Timing.{fmt}"),
                 save_format=fmt)
    plt.close(fig)


def fig_ablation_cost(df: pd.DataFrame, out: str, fmt: str = "pdf") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for algo in df["algorithm"].unique():
        sub = df_clean[df_clean["algorithm"] == algo]
        g = sub.groupby("n_customers")["avg_cost"].agg(["mean", "std"]).reset_index().dropna()
        if g.empty:
            continue
        ax.errorbar(g["n_customers"], g["mean"], yerr=g["std"].fillna(0),
                    marker=MARKERS.get(algo, "o"),
                    color=COLORS_ABLATION.get(algo, "#333"),
                    label=ABLATION_DISPLAY_NAMES.get(algo, algo),
                    markersize=8, capsize=3, linewidth=3 if algo == "SA-ALNS" else 2)
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Average penalised cost (\\$)")
    ax.set_title("Ablation study: penalised cost vs instance size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(out, "part2_ablation", f"Figure_Ablation_Cost.{fmt}"),
                 save_format=fmt)
    plt.close(fig)


def fig_multicomp_AB(df: pd.DataFrame, out: str, fmt: str = "pdf") -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    clean = df.replace([np.inf, -np.inf], np.nan)

    gA = clean.groupby("n_customers")["A_full_cost"].mean()
    gB = clean.groupby("n_customers")["B_nomix_cost"].mean()
    axes[0].plot(gA.index, gA.values, marker="o", color="#2E86AB",
                 label="A: full (multi-product)", linewidth=2)
    axes[0].plot(gB.index, gB.values, marker="s", color="#E76F51",
                 label="B: single-product", linewidth=2)
    axes[0].set_xlabel("Number of customers")
    axes[0].set_ylabel("Mean penalised cost (\\$)")
    axes[0].set_title("Cost: multi-compartment benefit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    gA = clean.groupby("n_customers")["A_full_fleet"].mean()
    gB = clean.groupby("n_customers")["B_nomix_fleet"].mean()
    axes[1].plot(gA.index, gA.values, marker="o", color="#2E86AB", label="A: full", linewidth=2)
    axes[1].plot(gB.index, gB.values, marker="s", color="#E76F51", label="B: single-product", linewidth=2)
    axes[1].set_xlabel("Number of customers")
    axes[1].set_ylabel("Mean fleet size")
    axes[1].set_title("Fleet size: multi-compartment benefit")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    data = clean["cost_improvement_AB"].dropna()
    if len(data):
        axes[2].hist(data, bins=15, color="#2E86AB", edgecolor="black", alpha=0.8)
        axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5, label="No benefit")
        axes[2].axvline(data.mean(), color="black", linewidth=1.5,
                        label=f"Mean = {data.mean():+.1f}\\%")
        axes[2].set_xlabel("Cost improvement A over B (\\%)")
        axes[2].set_ylabel("Instances")
        axes[2].set_title("Distribution of multi-compartment benefit")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Multi-compartment benefit", fontsize=12)
    plt.tight_layout()
    safe_savefig(fig, os.path.join(out, "part2_ablation", f"Figure_Multicomp_AB.{fmt}"),
                 save_format=fmt)
    plt.close(fig)


def fig_scalability_ratio(df: pd.DataFrame, out: str, fmt: str = "pdf") -> None:
    if df.empty:
        return
    clean = df.replace([np.inf, -np.inf], np.nan)
    pivot = clean.pivot_table(index="n_customers", columns="algorithm",
                               values="avg_cost", aggfunc="mean")
    if "Standard_ALNS" not in pivot.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in pivot.columns:
        if algo == "Standard_ALNS":
            continue
        ratio = (pivot[algo] / pivot["Standard_ALNS"]).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            continue
        ax.plot(ratio.index, ratio.values, marker=MARKERS.get(algo, "o"),
                color=COLORS_SOTA.get(algo, "#333"),
                label=SOTA_DISPLAY_NAMES.get(algo, algo),
                linewidth=3 if algo == "SA-ALNS" else 2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="Parity with Standard ALNS")
    ax.set_xlabel("Number of customers")
    ax.set_ylabel("Competitor / Standard ALNS cost ratio")
    ax.set_title("Cost ratio to Standard ALNS by instance size")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(out, "part1_sota", f"Figure_Scalability_Ratio.{fmt}"),
                 save_format=fmt)
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

SOTA_ALGORITHMS = {
    "SA-ALNS": SAALNS,
    "SA-ALNS_CostOnly": SAALNSNoStability,
    "Standard_ALNS": StandardALNS,
    "FDAHS": FDAHSAlgorithm,
    "GCOF": GCOFAlgorithm,
    "HSA": HSAAlgorithm,
    "ACO_2opt": ACO2optAlgorithm,
}

ABLATION_ALGORITHMS = {
    "SA-ALNS": SAALNS,
    "SA-ALNS_NS": SAALNSNoStability,
    "SA-ALNS_NA": SAALNSNoAdaptive,
    "SA-ALNS_SEQ": SAALNSSequential,
    "SA-ALNS_NC": SAALNSNoCache,
    "SA-ALNS_IMB": SAALNSNoImbalance,
}


def test_cache_key_order_sensitivity() -> None:
    assert SACA.key([0, 1]) != SACA.key([1, 0])
    assert SACA.key([0, 1, 2]) != SACA.key([2, 1, 0])
    assert SACA.key([0, 1, 2]) == SACA.key([0, 1, 2])
    print("  [test] test_cache_key_order_sensitivity: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="DSC-CTRP v8.9.4 reference implementation")
    ap.add_argument("--max-instances", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--supplementary", action="store_true",
                    help="Emit demoted figures (CG, LDD, ablation-safety, "
                         "compartment-fleet) to supplementary/.")
    ap.add_argument("--format", choices=("pdf", "png"), default="pdf")
    ap.add_argument("--skip-instances", action="store_true")
    ap.add_argument("--skip-scalability", action="store_true")
    args = ap.parse_args()

    print("=" * 80)
    print("DSC-CTRP v8.9.4")
    print("=" * 80)
    print(f"  Base directory  : {BASE_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Max instances   : {args.max_instances}")
    print(f"  Seeds           : {args.seeds}")
    print(f"  Supplementary   : {args.supplementary}")
    print(f"  Figure format   : {args.format}")

    test_cache_key_order_sensitivity()
    if not args.skip_instances:
        ensure_instances(BASE_DIR)

    instances = _list_instances(BASE_DIR)[: args.max_instances]

    sota_df = run_sota(instances, OUTPUT_DIR, args.seeds, SOTA_ALGORITHMS)
    sota_df.to_csv(os.path.join(OUTPUT_DIR, "sota_comparison_results.csv"), index=False)

    if args.seeds == 1:
        run_statistics(sota_df, OUTPUT_DIR)

    ablation_df = run_sota(instances, OUTPUT_DIR, 1, ABLATION_ALGORITHMS)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_study_results.csv"), index=False)
    paired_ablation(OUTPUT_DIR)

    if args.seeds == 1:
        multicomp_df = run_multicomp(sota_df, BASE_DIR, OUTPUT_DIR)
        paired_multicomp(multicomp_df, OUTPUT_DIR)
    else:
        multicomp_df = pd.DataFrame()

    run_sensitivity(BASE_DIR, OUTPUT_DIR, max_instances=min(5, args.max_instances))

    scalability_df = pd.DataFrame()
    if args.seeds == 1 and not args.skip_scalability:
        scalability_df = run_scalability(BASE_DIR, OUTPUT_DIR, SOTA_ALGORITHMS)

    if not sota_df.empty:
        fig_sota_cost(sota_df, OUTPUT_DIR, args.format)
        fig_sota_timing(sota_df, OUTPUT_DIR, args.format)
        fig_scalability_ratio(scalability_df, OUTPUT_DIR, args.format)
    if not ablation_df.empty:
        fig_ablation_cost(ablation_df, OUTPUT_DIR, args.format)
    if not multicomp_df.empty:
        fig_multicomp_AB(multicomp_df, OUTPUT_DIR, args.format)

    print("\n" + "=" * 80)
    print("FIGURE INTEGRITY CHECK")
    print("=" * 80)
    n_checked = n_corrupted = 0
    for d in (os.path.join(OUTPUT_DIR, "part1_sota"),
              os.path.join(OUTPUT_DIR, "part2_ablation")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".pdf", ".png")):
                continue
            n_checked += 1
            p = os.path.join(d, f)
            if f.lower().endswith(".pdf") and not verify_pdf_integrity(p):
                print(f"  [CORRUPTED] {p}")
                n_corrupted += 1
    if n_corrupted == 0:
        print(f"  All {n_checked} figures pass integrity check.")
    else:
        print(f"  {n_corrupted}/{n_checked} corrupted; rerun with --format png.")
    print("=" * 80)


if __name__ == "__main__":
    main()
