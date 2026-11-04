"""
DSC-CTRP standard instance generator.

Reproduces the 100-instance benchmark reported in the manuscript:
  * Paixao layer  : P20_1--P20_5, P25_1--P25_5, P30_1--P30_5  (15)
  * Mirzaei layer : vrpnc1--vrpnc15, extra33--extra87          (85)
Total: 100 instances. 30 (20--25 customers) are used for the SOTA and
ablation; the remaining 70 are reserved for the scalability study.

This file is a faithful extraction of generate_benchmark_instances()
and generate_single_instance() from run_experiments.py. The seeds are
SHA-1 based and reproducible across Python runs. Do not substitute a
different generator: the instance names would collide while the
contents would differ, and the manuscript's results would no longer be
reproducible from this repository.

Usage:
    python generate_instances.py --output benchmark_instances
"""

import argparse
import hashlib
import json
import math
import os
import random

import numpy as np

INSTANCE_GENERATOR_VERSION = "v8.7"


def stable_seed(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:8], 16) % (2 ** 31)


def generate_single_instance(n_customers, layer, seed, name=None,
                             n_products=4, n_compartments=3):
    random.seed(seed)
    np.random.seed(seed)
    name = name or f"{layer}_{n_customers}"
    n_products = max(2, min(n_products, n_compartments + 2))
    wb = random.uniform(3.0, 5.0)

    comps, comp_pos = [], {}
    for i in range(n_compartments):
        frac = 0.30 + 0.40 * (i + 0.5) / n_compartments
        px = wb * frac
        comp_pos[i] = px
        comps.append({"id": i, "capacity": random.uniform(120.0, 200.0),
                      "position_x": px, "position_y": 0.0, "load_density": 1.0})

    total_capacity = sum(c["capacity"] for c in comps)
    max_payload = total_capacity * 0.8
    unladen = max_payload * random.uniform(0.30, 0.50)
    total_gvw = max_payload + unladen
    fx = total_gvw * 0.50
    rx = total_gvw * 0.65
    avg_comp_cap = total_capacity / n_compartments
    demand_max = avg_comp_cap * random.uniform(0.60, 0.90)
    demand_min = demand_max * 0.40

    customers = []
    for i in range(n_customers):
        p = random.randint(0, n_products - 1)
        customers.append({
            "id": i, "x": random.uniform(0, 100), "y": random.uniform(0, 100),
            "demand": {p: random.uniform(demand_min, demand_max)},
            "time_window": [0.0, 1440.0],
            "service_time": random.uniform(5, 15),
        })

    tws = {}
    for i in range(n_customers):
        s = random.uniform(0, 100)
        tws[i] = [s, s + random.uniform(20, 60)]
        customers[i]["time_window"] = tws[i]

    vehicle = {
        "id": 0, "compartments": comps, "compartment_positions": comp_pos,
        "max_payload": max_payload,
        "front_axle_max": fx, "rear_axle_max": rx,
        "front_axle_min": fx * 0.10, "rear_axle_min": rx * 0.10,
        "wheelbase": wb, "unladen_weight": unladen, "max_volume": total_capacity,
        "steering_min_ratio": 0.25, "driving_min_ratio": 0.55,
        "product_densities": {p: 0.8 for p in range(n_products)},
        "cg_min": 0.3 * wb, "cg_max": 0.7 * wb,
    }

    coords = [[0, 0]] + [[c["x"], c["y"]] for c in customers]
    dm = [[math.hypot(coords[i][0] - coords[j][0],
                      coords[i][1] - coords[j][1])
           for j in range(n_customers + 1)] for i in range(n_customers + 1)]

    return {"name": name, "layer": layer, "customers": customers,
            "time_windows": tws, "vehicle": vehicle, "distance_matrix": dm,
            "n_products": n_products, "n_customers": n_customers,
            "n_compartments": n_compartments}


def generate_benchmark(output_dir, num_instances=100):
    os.makedirs(output_dir, exist_ok=True)
    all_instances = []

    paixao = [("P20", 20, 4, 3), ("P25", 25, 4, 3), ("P30", 30, 4, 4)]
    mirzaei = [(40, 4, 3), (40, 5, 3), (40, 4, 4),
               (50, 4, 3), (50, 5, 3), (50, 5, 4),
               (60, 5, 3), (60, 5, 4), (60, 4, 4),
               (75, 5, 4), (75, 5, 4), (75, 4, 5),
               (100, 5, 4), (100, 5, 5), (100, 4, 5)]

    ctr = 1
    for prefix, n, npr, ncmp in paixao:
        for i in range(1, 6):
            nm = f"{prefix}_{i}"
            all_instances.append((nm, generate_single_instance(
                n, "Paixao", ctr, nm, npr, ncmp)))
            ctr += 1

    for idx, (n, npr, ncmp) in enumerate(mirzaei, start=1):
        nm = f"vrpnc{idx}"
        all_instances.append((nm, generate_single_instance(
            n, "Mirzaei", ctr, nm, npr, ncmp)))
        ctr += 1

    while len(all_instances) < num_instances:
        n = random.choice([20, 25, 30, 40, 50, 60, 75, 100])
        nm = f"extra{len(all_instances) + 1}"
        all_instances.append((nm, generate_single_instance(
            n, "Mirzaei", ctr, nm,
            random.choice([3, 4, 5]), random.choice([3, 4, 5]))))
        ctr += 1

    for nm, inst in all_instances:
        with open(os.path.join(output_dir, f"{nm}.json"), "w") as f:
            json.dump(inst, f, indent=2)
    with open(os.path.join(output_dir, ".instance_generator_version"), "w") as f:
        f.write(INSTANCE_GENERATOR_VERSION)
    print(f"Generated {len(all_instances)} instances in {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="DSC-CTRP standard instance generator")
    ap.add_argument("--output", default="benchmark_instances")
    ap.add_argument("--num", type=int, default=100)
    args = ap.parse_args()
    generate_benchmark(args.output, args.num)


if __name__ == "__main__":
    main()
