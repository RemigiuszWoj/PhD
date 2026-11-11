"""E1 anatomy of the crossing paper (articles/crossings/PLAN.md, E1 with the v3 note).

States are sampled from 10 s ILS runs of the plain variants cross_dyn, cross_c0 and cross_inv (seed 0),
uniformly over each run's iterations (reservoir sampling), so no variant picks its own convenient states.
Each state is analysed with the campaign's candidate set (block filter, top-K caps) and selection code:
  - predicted and real gain of every family (dyn, c0, c1, pa1, inv), swaps, crossings, nestings and the
    largest number of crossings on one arc;
  - the offline counter that replaces the safeguard: is the plain compound move worse than, equal to or
    better than the best single swap (whose delta is exact), and does it improve at all;
  - the smallest c with OPT_c = OPT_inv (0, 1 or "2+"), from the DP values of c0 and c1;
  - additivity of random pairs of improving swaps, a third per geometry (disjoint, nested, crossing),
    with the length of the overlap of the two intervals for the regression in E1.
QUBO couplers and the second-order model stay in pilot/pilot_qubo_structure.py and pilot/pilot_pairq.py.

Usage (repo root):
    python scripts/crossing_anatomy.py --dry-run
    python scripts/crossing_anatomy.py --files tai20_5 --limit 2            # smoke test
    taskpolicy -b python scripts/crossing_anatomy.py --workers 4             # next to another campaign
Output: results/crossings/anatomy/<file>__inst=<i>.json, one file per instance (the run resumes).
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import src.algorithms.ils as ils_mod  # noqa: E402
from src.neighborhoods.classical.crossing import (  # noqa: E402
    crossing_neighborhood, crossings, max_crossings_per_arc, nestings, parse_crossing_mode,
)
from src.neighborhoods.classical.kbonacci import compute_candidates  # noqa: E402
from src.parser import parser  # noqa: E402
from src.permutation_procesing import c_max  # noqa: E402

TAILLARD = [
    "tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20",
    "tai100_5", "tai100_10", "tai100_20", "tai200_10", "tai200_20",
]   # tai500_20 skipped: one-move regime in 10 s
INSTANCES = list(range(10))
SOURCES = ["cross_dyn", "cross_c0", "cross_inv"]       # v3: plain variants only
FAMILIES = ["dyn", "c0", "c1", "pa1", "inv"]
OUT = ROOT / "results" / "crossings" / "anatomy"


def sample_states(p, mode, time_ms, k, rng):
    """One ILS run as in the campaign (identity start, tabu 10, mushroom 10, seed 0) that keeps k states
    drawn uniformly from its iterations. The spy only copies pi, it does not touch the search."""
    orig = ils_mod.get_neighbor
    seen, kept = [0], []

    def spy(neigh_mode, pi, *args, **kwargs):
        t = seen[0]
        seen[0] += 1
        if len(kept) < k:
            kept.append((t, pi.copy()))
        else:
            r = rng.randrange(t + 1)
            if r < k:
                kept[r] = (t, pi.copy())
        return orig(neigh_mode, pi, *args, **kwargs)

    ils_mod.get_neighbor = spy
    try:
        random.seed(0)
        _, best, _, _, _ = ils_mod.iterated_local_search(
            p, max_time_ms=time_ms, tabu_tenure=10, neigh_mode=mode, mushroom_k=10, diversification="mushroom")
    finally:
        ils_mod.get_neighbor = orig
    return sorted(kept), seen[0], best


def geometry(x, y):
    (a, b), (c, d) = sorted([x, y])
    if b < c:
        return "disjoint", 0
    if c < b < d:
        return "crossing", b - c
    return "nested", d - c          # a < c < d < b


def analyse(pi, p, boundary, rng, n_pairs):
    base, cand = compute_candidates(pi, p, max(2, len(pi)), True, boundary)
    improving = {(i, j): d for i, j, d in cand if d < 0}
    best1 = min((d for _, _, d in cand), default=0)
    rec = {"base": base, "n_cand": len(cand), "n_improving": len(improving), "best_single": best1}
    if not improving:
        return rec
    fam = {}
    for f in FAMILIES:
        st = {}
        _, _, arcs = crossing_neighborhood(pi, p, f, "plain", boundary, stats=st)
        real = st["real_delta"]
        fam[f] = {"pred": st["predicted_delta"], "real": real, "swaps": len(arcs), "cr": crossings(arcs),
                  "nest": nestings(arcs), "maxcr_arc": max_crossings_per_arc(arcs),
                  "cap_top": st["cap_top"], "cap_core": st["cap_core"],
                  "vs_single": "worse" if real > best1 else ("equal" if real == best1 else "better"),
                  "improves": real < 0}
    rec["fam"] = fam
    opt = fam["inv"]["pred"]
    rec["min_c"] = 0 if abs(fam["c0"]["pred"] - opt) < 1e-9 else (1 if abs(fam["c1"]["pred"] - opt) < 1e-9 else "2+")
    arcs = sorted(improving)
    pairs = {"disjoint": [], "nested": [], "crossing": []}
    tries = 0
    while len(arcs) >= 2 and tries < 20 * n_pairs and min(len(v) for v in pairs.values()) < n_pairs // 3:
        tries += 1
        x, y = rng.sample(arcs, 2)
        if len({*x, *y}) < 4:
            continue
        g, overlap = geometry(x, y)
        if len(pairs[g]) >= n_pairs // 3:
            continue
        q = pi.copy()
        for a, b in (x, y):
            q[a], q[b] = q[b], q[a]
        pairs[g].append([improving[x] + improving[y], c_max(q, p) - base, overlap])
    rec["pairs"] = pairs            # [predicted, real, overlap] per pair
    return rec


def job(args):
    file, inst, time_ms, k, n_pairs = args
    out = OUT / f"{file}__inst={inst}.json"
    if out.exists():
        return out.name, "skip"
    t0 = time.perf_counter()
    p = parser(f"data/{file}.txt", inst)["processing_times"]
    rng = random.Random(f"{file}/{inst}")       # separate stream: the ILS keeps its own global one
    states = []
    for mode in SOURCES:
        family, rule, boundary = parse_crossing_mode(mode)
        kept, iters, best = sample_states(p, mode, time_ms, k, rng)
        for t, pi in kept:
            rec = analyse(pi, p, boundary, rng, n_pairs)
            rec.update({"source": mode, "iter": t, "run_iters": iters, "run_best": best})
            states.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    json.dump({"file": file, "inst": inst, "time_ms": time_ms, "states": states}, open(tmp, "w"))
    tmp.rename(out)
    return out.name, f"{len(states)} states, {time.perf_counter() - t0:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=",".join(TAILLARD))
    ap.add_argument("--instances", default=",".join(map(str, INSTANCES)))
    ap.add_argument("--time-ms", type=int, default=10000, help="ILS budget per source run")
    ap.add_argument("--states", type=int, default=7, help="states kept per source run (3 sources ~ 20 per instance)")
    ap.add_argument("--pairs", type=int, default=300, help="pairs per state, a third per geometry")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="at most N instances (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    jobs = [(f, i, a.time_ms, a.states, a.pairs) for f in a.files.split(",") for i in map(int, a.instances.split(","))]
    if a.limit:
        jobs = jobs[:a.limit]
    todo = [j for j in jobs if not (OUT / f"{j[0]}__inst={j[1]}.json").exists()]
    print(f"{len(jobs)} instances, {len(todo)} to do, {len(SOURCES)} ILS runs of {a.time_ms / 1000:g} s each, "
          f"{a.workers} worker(s)")
    if a.dry_run:
        return
    with Pool(a.workers) as pool:
        for name, msg in pool.imap_unordered(job, todo):
            print(f"[anatomy] {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
