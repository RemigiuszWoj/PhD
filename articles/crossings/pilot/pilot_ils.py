"""ILS pilot (repo ILS unchanged, tenure 10, mushroom 10, start = identity) with new crossing-family modes.

Modes cr_<v>: all-pairs deltas (Head+Tail, no filter = lossless), improving candidates, selection:
  c0, c1, pa1, inv          additive exact selection
  c0_safe, inv_safe          additive + verify: if the compound move is worse than the best single swap, take the single swap
  pq_c0, pq_inv              pair-corrected model (top-K = 40 candidates), exact MILP
Reference: repo 'motzkin' (old code, last-machine filter) and 'dynasearch' (old code).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "results"
import json, random, sys, time
from multiprocessing import Pool
import src.algorithms.ils as ils_mod
from src.algorithms.ils import iterated_local_search
from src.permutation_procesing import c_max
from pilot_anatomy import lcg_instance, all_deltas, apply
from crossing_select import select_c0, select_c1, select_perarc1, select_inv
from pilot_pairq import pair_qubo_milp
from taillard_lb import lb

_orig = ils_mod.get_neighbor
SEL = {"c0": select_c0, "c1": select_c1, "pa1": select_perarc1, "inv": select_inv}
STATS = {"moves": 0, "cr": 0, "swaps": 0, "worse_than_single": 0}

def cr_neighbor(variant, pi, p):
    base, D = all_deltas(pi, p)
    C = {ab: w for ab, w in D.items() if w < 0}
    single = min(D, key=D.get)
    if not C:
        mv = [single]
    elif variant == "swap":
        mv = [single]
    elif variant.startswith("pq"):
        K = int(variant.split("_")[0][2:] or 40)
        top = dict(sorted(C.items(), key=lambda kv: kv[1])[:K])
        arcs = sorted(top); Q = {}
        for x in range(len(arcs)):
            for y in range(x + 1, len(arcs)):
                a, b = arcs[x], arcs[y]
                if len({*a, *b}) < 4: continue
                Q[(a, b)] = (c_max(apply(pi, [a, b]), p) - base) - top[a] - top[b]
        _, mv = pair_qubo_milp(top, Q, forbid_cross=variant.endswith("_c0"))
        if not mv: mv = [single]
    else:
        key = variant.replace("_safe", "")
        _, mv = SEL[key](C)
        if not mv: mv = [single]
    new_pi = apply(pi, mv); new_c = c_max(new_pi, p)
    STATS["moves"] += 1
    if variant.endswith("_safe") and len(mv) > 1:
        sp = apply(pi, [single]); sc = c_max(sp, p)
        if sc < new_c:
            new_pi, new_c, mv = sp, sc, [single]; STATS["worse_than_single"] += 1
    return new_pi, new_c, tuple(sorted(mv))

def patched(neigh_mode, current_pi, processing_times, n, tabu_len=None, quantum_config=None):
    if neigh_mode.startswith("cr_"):
        new_pi, new_c, mid = cr_neighbor(neigh_mode[3:], current_pi, processing_times)
        return new_pi, new_c, mid, None
    return _orig(neigh_mode, current_pi, processing_times, n, tabu_len, quantum_config)
ils_mod.get_neighbor = patched

TAI = {"tai20_5": (20, 5, [873654221, 379008056, 1866992158, 216771124]),
       "tai20_10": (20, 10, [587595453, 1401007982, 873136276, 268827376]),
       "lcg50_10": (50, 10, [1958948863, 575633267, 655816003, 1977864101])}
MODES = ["motzkin", "dynasearch", "cr_c0", "cr_c1", "cr_pa1", "cr_inv", "cr_c0_safe", "cr_inv_safe", "cr_pq_c0", "cr_pq_inv"]
if len(sys.argv) > 2: MODES = sys.argv[2].split(",")

def run(args):
    sys.setrecursionlimit(100000)
    f, idx, seed, mode, tl = args
    n, m, seeds = TAI[f]; p = lcg_instance(seeds[idx], n, m)
    random.seed(seed)
    for k in STATS: STATS[k] = 0
    best_pi, best_c, _, _, st = iterated_local_search(p, max_time_ms=tl, tabu_tenure=10, neigh_mode=mode, mushroom_k=10)
    L = lb(p)
    return {"file": f, "inst": idx, "seed": seed, "mode": mode, "tl": tl, "cmax": best_c, "lb": L,
            "rpd": 100 * (best_c - L) / L, "iters": st.get("iterations"),
            "fallback_share": STATS["worse_than_single"] / max(1, STATS["moves"])}

if __name__ == "__main__":
    tl = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    jobs = [(f, i, s, mode, tl) for f in TAI for i in range(4) for s in range(3) for mode in MODES]
    random.Random(5).shuffle(jobs)
    with Pool(3) as pool:
        res = list(pool.imap_unordered(run, jobs))
    json.dump(res, open(RES / f"ils_{tl}ms_{len(MODES)}modes.json", "w"))
    import statistics as st
    from collections import defaultdict
    g = defaultdict(list)
    for r in res: g[(r["file"], r["mode"])].append(r)
    print(f"ILS {tl} ms, RPD [%] to LB (mean over 4 inst x 3 seeds), mean iterations")
    print("mode         " + "  ".join(f"{f:>18}" for f in TAI))
    for mode in MODES:
        print(f"{mode:12s} " + "  ".join(f"{st.mean(r['rpd'] for r in g[(f, mode)]):8.2f} ({st.mean(r['iters'] for r in g[(f, mode)]):6.0f} it, fb {st.mean(r['fallback_share'] for r in g[(f, mode)]):.2f})" for f in TAI))
