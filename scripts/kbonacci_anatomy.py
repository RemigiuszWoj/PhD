"""Anatomy of the k-bonacci move on search states (no ILS/SA involved).

For every instance a few descents are run from random permutations with a
reference k (real improvement only). States are sampled along each descent
(start, early, middle, local optimum). On every state and for every k the best
compound move is computed with both candidate filters and the script records
    predicted delta (sum of deltas) and real delta  -> additivity error vs k
    distances of the selected swaps                 -> how far the DP reaches
    number of candidates and evaluation time        -> cost vs k
    number of critical-path boundaries |U|          -> block length n / (|U| + 1)
    k_sat = smallest k with >= 95% of the real gain of the largest k

The states are identical for all k, so every comparison is paired.

Usage:
    python scripts/kbonacci_anatomy.py                       # n <= 100, ~1 h on one core
    python scripts/kbonacci_anatomy.py --files tai20_5 --instances 2 --starts 2   # smoke
    python scripts/kbonacci_anatomy.py --files tai200_10,tai200_20 --ks 2,3,4,6,8,12,16
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neighborhoods.accelerator import compute_critical_path_boundaries  # noqa: E402
from src.neighborhoods.classical.kbonacci import kbonacci_neighborhood  # noqa: E402
from src.neighborhoods.common import compute_head  # noqa: E402
from src.parser import parser  # noqa: E402
from src.permutation_procesing import c_max  # noqa: E402

FILES = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20",
         "tai100_5", "tai100_10", "tai100_20"]
KS = ["2", "3", "4", "5", "6", "8", "12", "16", "n"]
FILTERS = ["last_machine", "critical_path"]
K_REF = 8
SAMPLE_STEPS = (0, 2, 5)


def descent_states(pt, pi, max_steps=10_000):
    """States of a real-improvement descent with k = K_REF: selected steps + local optimum."""
    states, step, cur = [], 0, c_max(pi, pt)
    while step < max_steps:
        if step in SAMPLE_STEPS:
            states.append(("step%d" % step, pi))
        new_pi, new_c, _ = kbonacci_neighborhood(pi, pt, K_REF, force_move_if_none=False,
                                                 boundary_mode="critical_path")
        if new_c >= cur:
            break
        pi, cur, step = new_pi, new_c, step + 1
    states.append(("local_opt", pi))
    return states


def job(args):
    f, inst, starts, ks = args
    d = parser(f"data/{f}.txt", inst)
    pt = d["processing_times"]
    m, n = len(pt), len(pt[0])
    rng = random.Random(7000 + inst)
    rows = []
    for s in range(starts):
        pi = list(range(n))
        rng.shuffle(pi)
        for label, state in descent_states(pt, pi):
            U = compute_critical_path_boundaries(compute_head(state, pt), pt, state)
            for flt in FILTERS:
                for k_label in ks:
                    k = n if k_label == "n" else int(k_label)
                    if k > n:
                        continue
                    st: dict = {}
                    t0 = time.perf_counter()
                    kbonacci_neighborhood(state, pt, k, force_move_if_none=False, stats=st,
                                          boundary_mode=flt)
                    rows.append({
                        "file": f, "n": n, "m": m, "inst": inst, "start": s, "state": label,
                        "filter": flt, "k": k_label, "U": len(U),
                        "t_ms": 1000 * (time.perf_counter() - t0),
                        "cand": st.get("num_candidates", 0), "swaps": st.get("num_swaps", 0),
                        "pred": st.get("predicted_delta", 0), "real": st.get("real_delta", 0),
                        "dist": st.get("distances", []),
                    })
    return rows


def summarize(rows, ks) -> None:
    def sel(**kw):
        return [r for r in rows if all(r[a] == b for a, b in kw.items())]

    for flt in FILTERS:
        print(f"\nfilter = {flt}: mean over states")
        print(f"{'file':<10}{'k':>4}{'ms':>9}{'cand':>8}{'swaps':>7}{'pred':>9}{'real':>9}"
              f"{'exact':>7}{'optim.':>8}{'q50 d':>7}{'q90 d':>7}")
        for f in sorted({r["file"] for r in rows}, key=lambda x: tuple(int(v) for v in x[3:].split("_"))):
            for k in ks:
                rs = sel(file=f, filter=flt, k=k)
                if not rs:
                    continue
                dist = sorted(x for r in rs for x in r["dist"])
                q = (lambda p: dist[min(len(dist) - 1, int(p * len(dist)))]) if dist else (lambda p: 0)
                mean = lambda key: statistics.mean(r[key] for r in rs)  # noqa: E731
                print(f"{f:<10}{k:>4}{mean('t_ms'):>9.2f}{mean('cand'):>8.1f}{mean('swaps'):>7.2f}"
                      f"{mean('pred'):>9.1f}{mean('real'):>9.1f}"
                      f"{sum(r['pred'] == r['real'] for r in rs) / len(rs):>7.2f}"
                      f"{sum(r['real'] > r['pred'] for r in rs) / len(rs):>8.2f}{q(0.5):>7}{q(0.9):>7}")

    # k_sat per state (critical-path filter = the exact neighborhood)
    print("\nk_sat (smallest k with >= 95% of the real gain of the largest k), critical-path filter")
    print(f"{'file':<10}{'states':>8}{'mean |U|':>10}{'n/(|U|+1)':>11}{'median k_sat':>14}")
    by_state = {}
    for r in sel(filter="critical_path"):
        by_state.setdefault((r["file"], r["inst"], r["start"], r["state"]), {})[r["k"]] = r
    per_file = {}
    for (f, *_), d in by_state.items():
        order = [k for k in ks if k in d]
        gain_max = -d[order[-1]]["real"]
        if gain_max <= 0:
            continue
        k_sat = next(k for k in order if -d[k]["real"] >= 0.95 * gain_max)
        r0 = d[order[0]]
        per_file.setdefault(f, []).append((r0["n"] if k_sat == "n" else int(k_sat), r0["U"], r0["n"]))
    for f, v in per_file.items():
        mean_u = statistics.mean(x[1] for x in v)
        print(f"{f:<10}{len(v):>8}{mean_u:>10.1f}{v[0][2] / (mean_u + 1):>11.1f}"
              f"{statistics.median(x[0] for x in v):>14}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", default=",".join(FILES))
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--ks", default=",".join(KS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="articles/kbonacci/anatomy/anatomy.json")
    a = ap.parse_args()
    ks = a.ks.split(",")
    jobs = [(f, i, a.starts, ks) for f in a.files.split(",") for i in range(a.instances)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.map(job, jobs) for r in part]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh)
    print(f"{len(rows)} rows -> {a.out}")
    summarize(rows, ks)


if __name__ == "__main__":
    main()
