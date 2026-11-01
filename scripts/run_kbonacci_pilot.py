"""Pilot study of the k-bonacci neighborhood family (Phase 1 of articles/kbonacci/PLAN.md).

Two experiments, both CPU-only:
  1. ILS with k-bonacci neighborhoods, k in K_VALUES, on a few Taillard instances
     -> RPD to the upper bound stored in the Taillard file, iterations in budget.
  2. Neighborhood anatomy on random permutations: candidates, evaluation time,
     swaps per move, and the additivity error |predicted - real delta|.

Usage:
    python scripts/run_kbonacci_pilot.py [--tl 1000] [--seeds 3] [--out results/kbonacci_pilot]
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

sys.setrecursionlimit(1_000_000)  # dynasearch_full uses a recursive DP (depth ~ #candidates)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.ils import iterated_local_search  # noqa: E402
from src.neighborhoods.classical.kbonacci import kbonacci_neighborhood  # noqa: E402
from src.parser import parser  # noqa: E402

FILES = ["data/tai20_5.txt", "data/tai20_10.txt", "data/tai50_10.txt"]
INSTANCES = [0, 1, 2, 3]
K_VALUES = [
    "fibonacci",
    "kbonacci_k2",
    "tribonacci",
    "kbonacci_k4",
    "kbonacci_k6",
    "kbonacci_k8",
    "kbonacci_kn",
    "dynasearch",
    "adjacent",
]


def _ils_job(args):
    f, inst, mode, seed, tl = args
    d = parser(f, inst)
    random.seed(seed)
    t0 = time.perf_counter()
    _, best_c, _, _, st = iterated_local_search(
        d["processing_times"], max_time_ms=tl, neigh_mode=mode
    )
    ub = d["info"]["upper_bound"]
    return {
        "file": os.path.basename(f),
        "inst": inst,
        "mode": mode,
        "seed": seed,
        "tl": tl,
        "cmax": best_c,
        "ub": ub,
        "rpd_ub": 100.0 * (best_c - ub) / ub,
        "iterations": st.get("iterations"),
        "wall_s": time.perf_counter() - t0,
    }


def _anatomy_job(args):
    f, inst, k_label, samples = args
    d = parser(f, inst)
    pt = d["processing_times"]
    n = len(pt[0])
    k = n if k_label == "n" else int(k_label)
    rng = random.Random(1000 + inst)
    rows = []
    for _ in range(samples):
        pi = list(range(n))
        rng.shuffle(pi)
        st: dict = {}
        t0 = time.perf_counter()
        kbonacci_neighborhood(pi, pt, k, stats=st)
        rows.append(
            {
                "t_ms": 1000 * (time.perf_counter() - t0),
                "cand": st["num_candidates"],
                "swaps": st["num_swaps"],
                "pred": st["predicted_delta"],
                "real": st["real_delta"],
                "dist": st["distances"],
            }
        )

    def mean(key):
        return statistics.mean(r[key] for r in rows)

    add_err = [abs(r["pred"] - r["real"]) for r in rows]
    return {
        "file": os.path.basename(f),
        "inst": inst,
        "k": k_label,
        "samples": samples,
        "eval_ms": mean("t_ms"),
        "candidates": mean("cand"),
        "swaps": mean("swaps"),
        "pred_delta": mean("pred"),
        "real_delta": mean("real"),
        "add_err_mean": statistics.mean(add_err),
        "add_exact_share": sum(e == 0 for e in add_err) / samples,
        "mean_distance": (
            statistics.mean(x for r in rows for x in r["dist"])
            if any(r["dist"] for r in rows)
            else 0.0
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", default="results/kbonacci_pilot")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = [
        (f, i, mode, s, a.tl)
        for f in FILES
        for i in INSTANCES
        for mode in K_VALUES
        for s in range(a.seeds)
    ]
    anat = [
        (f, i, k, a.samples)
        for f in FILES
        for i in INSTANCES[:2]
        for k in ["2", "3", "4", "6", "8", "n"]
    ]
    with Pool(a.workers) as pool:
        ils_rows = pool.map(_ils_job, jobs)
        anat_rows = pool.map(_anatomy_job, anat)

    with open(os.path.join(a.out, "ils.json"), "w") as fh:
        json.dump(ils_rows, fh, indent=1)
    with open(os.path.join(a.out, "anatomy.json"), "w") as fh:
        json.dump(anat_rows, fh, indent=1)

    print(
        f"\nILS, tl={a.tl} ms, {len(INSTANCES)} inst x {a.seeds} seeds; mean RPD to UB [%] / iterations"
    )
    print(f"{'mode':<14}" + "".join(f"{os.path.basename(f)[:-4]:>22}" for f in FILES))
    for mode in K_VALUES:
        cells = []
        for f in FILES:
            r = [x for x in ils_rows if x["mode"] == mode and x["file"] == os.path.basename(f)]
            cells.append(
                f"{statistics.mean(x['rpd_ub'] for x in r):7.2f} / {statistics.mean(x['iterations'] for x in r):8.0f}"
            )
        print(f"{mode:<14}" + "".join(f"{c:>22}" for c in cells))

    print("\nAnatomy on random permutations (mean over instances):")
    print(
        f"{'file':<10}{'k':>4}{'eval ms':>10}{'cand':>8}{'swaps':>7}{'pred':>9}{'real':>9}{'|err|':>8}{'exact':>7}{'dist':>6}"
    )
    for r in anat_rows:
        print(
            f"{r['file'][:-4]:<10}{r['k']:>4}{r['eval_ms']:>10.2f}{r['candidates']:>8.1f}{r['swaps']:>7.2f}"
            f"{r['pred_delta']:>9.1f}{r['real_delta']:>9.1f}{r['add_err_mean']:>8.1f}{r['add_exact_share']:>7.2f}{r['mean_distance']:>6.2f}"
        )


if __name__ == "__main__":
    main()
