"""Pair-corrected selection vs additive selection on trajectory states.

QUBO-style model: min sum_a d_a x_a + sum_{a~b} Q_ab x_a x_b, Q_ab = real(a,b) - d_a - d_b
for compatible pairs (no shared endpoint), solved exactly as MILP (linearized products),
under involution constraints (any crossing) or Motzkin constraints (no crossing).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "results"
import json, random, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from src.permutation_procesing import c_max
from pilot_anatomy import lcg_instance, all_deltas, apply
from crossing_select import select_c0, select_c1, select_inv, crossings

def pair_qubo_milp(C, Q, forbid_cross):
    arcs = sorted(C); A = len(arcs); ix = {a: t for t, a in enumerate(arcs)}
    pairs = [(ix[a], ix[b], q) for (a, b), q in Q.items() if q != 0 and not (forbid_cross and is_cross(a, b))]
    nv = A + len(pairs)
    obj = np.zeros(nv); obj[:A] = [C[a] for a in arcs]
    for t, (_, _, q) in enumerate(pairs): obj[A + t] = q
    rows, lb, ub = [], [], []
    pts = sorted({p for ab in arcs for p in ab})
    for p in pts:
        r = np.zeros(nv)
        for x, (a, b) in enumerate(arcs):
            if p in (a, b): r[x] = 1
        rows.append(r); lb.append(-np.inf); ub.append(1)
    if forbid_cross:
        for x in range(A):
            for y in range(x + 1, A):
                if is_cross(arcs[x], arcs[y]):
                    r = np.zeros(nv); r[x] = 1; r[y] = 1; rows.append(r); lb.append(-np.inf); ub.append(1)
    for t, (x, y, q) in enumerate(pairs):
        if q > 0:   # y >= x_a + x_b - 1
            r = np.zeros(nv); r[x] = 1; r[y] = 1; r[A + t] = -1; rows.append(r); lb.append(-np.inf); ub.append(1)
        else:       # y <= x_a, y <= x_b
            r = np.zeros(nv); r[A + t] = 1; r[x] = -1; rows.append(r); lb.append(-np.inf); ub.append(0)
            r = np.zeros(nv); r[A + t] = 1; r[y] = -1; rows.append(r); lb.append(-np.inf); ub.append(0)
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": 30})
    xs = res.x[:A] > 0.5
    return res.fun, [arcs[x] for x in range(A) if xs[x]]

def is_cross(x, y):
    (a, b), (c, d) = sorted([x, y])
    return a < c < b < d

def job(args):
    n, m, seed = args
    p = lcg_instance(seed, n, m)
    pi = list(range(n)); cur = c_max(pi, p); out = []
    for step in range(200):
        base, D = all_deltas(pi, p); C = {ab: w for ab, w in D.items() if w < 0}
        if not C: break
        if 2 <= len(C) <= 60:
            arcs = sorted(C); Q = {}
            for x in range(len(arcs)):
                for y in range(x + 1, len(arcs)):
                    a, b = arcs[x], arcs[y]
                    if len({*a, *b}) < 4: continue
                    Q[(a, b)] = (c_max(apply(pi, [a, b]), p) - base) - C[a] - C[b]
            rec = {"n": n, "m": m, "seed": seed, "step": step, "nC": len(C), "best_single": min(C.values())}
            for name, f in (("c0", select_c0), ("c1", select_c1), ("inv", select_inv)):
                v, mv = f(C); real = c_max(apply(pi, mv), p) - base
                rec[name] = {"pred": v, "real": real, "swaps": len(mv), "cr": crossings(mv)}
            for name, fc in (("pq_inv", False), ("pq_c0", True)):
                t0 = time.perf_counter(); v, mv = pair_qubo_milp(C, Q, fc); dt = time.perf_counter() - t0
                real = c_max(apply(pi, mv), p) - base if mv else 0
                rec[name] = {"pred": v, "real": real, "swaps": len(mv), "cr": crossings(mv), "t_ms": dt * 1e3}
            out.append(rec)
        v, mv = select_c0(C); q = apply(pi, mv); qc = c_max(q, p)
        if qc >= cur:
            ab = min(C, key=C.get); q = apply(pi, [ab]); qc = c_max(q, p)
        pi, cur = q, qc
    return out

if __name__ == "__main__":
    rng = random.Random(77)
    jobs = [(n, m, rng.randrange(1, 2**31 - 1)) for (n, m) in [(20, 5), (20, 10), (20, 20), (50, 5), (50, 10), (50, 20)] for _ in range(4)]
    with Pool(4) as pool:
        res = [r for ch in pool.imap_unordered(job, jobs) for r in ch]
    json.dump(res, open(RES / "pairq.json", "w"))
    import statistics as st
    from collections import defaultdict
    g = defaultdict(list)
    for r in res: g[r["n"]].append(r)
    for n, L in sorted(g.items()):
        print(f"n={n}: states={len(L)} |C| med={st.median(r['nC'] for r in L)}")
        for name in ("c0", "c1", "inv", "pq_c0", "pq_inv"):
            real = [r[name]["real"] for r in L]
            better_single = sum(1 for r in L if r[name]["real"] < r["best_single"])
            worse = sum(1 for r in L if r[name]["real"] >= 0)
            print(f"   {name:7s} mean real={st.mean(real):7.2f} mean pred={st.mean(r[name]['pred'] for r in L):7.2f} "
                  f"P(beats best single swap)={better_single/len(L):.2f} P(not improving)={worse/len(L):.2f} swaps={st.mean(r[name]['swaps'] for r in L):.1f} cr={st.mean(r[name]['cr'] for r in L):.1f}")
        print(f"   best single swap mean real={st.mean(r['best_single'] for r in L):7.2f}; pq_inv MILP t_med={st.median(r['pq_inv']['t_ms'] for r in L):.0f} ms")
