"""Pilot for the crossing family on Taillard-like PFSP instances (LCG generator, random seeds).

Per state: all endpoint-swap deltas (Head+Tail, repo code), improving candidates,
best move for c=0 (Motzkin), c=1, per-arc<=1, c=inf (matching) [+ c=2 MILP for small |C|]:
predicted vs real gain, #swaps, crossings; QUBO conflict counts; pair additivity by geometry.
Descent: from identity and random starts, apply the selected move (fallback: best single
swap if the compound move does not really improve) until no improving single swap.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "results"
import json, math, random, sys, time
from multiprocessing import Pool
from src.neighborhoods.common import compute_endpoint_swap_delta, compute_head, compute_tail
from src.permutation_procesing import c_max
from crossing_select import (select_c0, select_c1, select_perarc1, select_inv, select_milp,
                    crossings, max_cross_per_arc)

def lcg_instance(seed, n, m):
    a, b, c, M = 16807, 127773, 2836, 2**31 - 1
    s, out = seed, [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            k = s // b; s = a * (s % b) - k * c
            if s < 0: s += M
            out[i][j] = 1 + int(s / M * 99)
    return out

def all_deltas(pi, p):
    n, m = len(pi), len(p)
    H = compute_head(pi, p); T = compute_tail(pi, p); base = H[m - 1][n - 1]
    D = {}
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[(i, j)] = compute_endpoint_swap_delta(pi, i, j, H, T, p, base)
    return base, D

def apply(pi, arcs):
    q = pi[:]
    for a, b in arcs: q[a], q[b] = q[b], q[a]
    return q

SEL = {"c0": select_c0, "c1": select_c1, "pa1": select_perarc1, "inv": select_inv}

def geometry(x, y):
    (a, b), (c, d) = sorted([x, y])
    if b < c: return "disjoint"
    if c < b < d: return "crossing"
    return "nested"   # a < c < d < b

def analyse_state(pi, p, rng, with_pairs=True, with_c2=True):
    base, D = all_deltas(pi, p)
    C = {ab: w for ab, w in D.items() if w < 0}
    rec = {"n_improving": len(C), "base": base}
    # QUBO conflict structure among improving candidates
    arcs = sorted(C)
    shared = crossp = nestp = 0
    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if len({a, b, c, d}) < 4: shared += 1
            elif c < b < d: crossp += 1
            elif d < b: nestp += 1
    rec.update({"conf_shared": shared, "conf_cross": crossp, "pairs_nested": nestp})
    sel = {}
    for name, f in SEL.items():
        t0 = time.perf_counter(); v, mv = f(C) if C else (0.0, []); dt = time.perf_counter() - t0
        real = (c_max(apply(pi, mv), p) - base) if mv else 0
        sel[name] = {"pred": v, "real": real, "swaps": len(mv), "cr": crossings(mv),
                     "maxcr_arc": max_cross_per_arc(mv), "t_ms": dt * 1e3}
    if with_c2 and 0 < len(C) <= 120:
        t0 = time.perf_counter(); v, mv = select_milp(C, 2, time_limit=20); dt = time.perf_counter() - t0
        real = (c_max(apply(pi, mv), p) - base) if mv else 0
        sel["c2"] = {"pred": v, "real": real, "swaps": len(mv), "cr": crossings(mv), "t_ms": dt * 1e3}
    rec["sel"] = sel
    if with_pairs and len(arcs) >= 2:
        pairs = {"disjoint": [], "nested": [], "crossing": []}
        tries = 0
        while tries < 600 and min(len(v) for v in pairs.values()) < 60:
            tries += 1
            x, y = rng.sample(arcs, 2)
            if len({*x, *y}) < 4: continue
            g = geometry(x, y)
            if len(pairs[g]) >= 60: continue
            real = c_max(apply(pi, [x, y]), p) - base
            pairs[g].append((C[x] + C[y], real))
        rec["pairs"] = {g: {"N": len(v), "sum_pred": sum(a for a, b in v), "sum_real": sum(b for a, b in v),
                            "nonimproving": sum(1 for a, b in v if b >= 0), "exact": sum(1 for a, b in v if a == b),
                            "sum_abs_err": sum(abs(a - b) for a, b in v)} for g, v in pairs.items()}
    return rec

def descent(pi, p, name, max_steps=400):
    f = SEL[name] if name in SEL else (lambda C: select_milp(C, 2, time_limit=20))
    steps = 0; t0 = time.perf_counter(); fallbacks = 0; tot_swaps = 0; tot_cr = 0
    cur = c_max(pi, p)
    while steps < max_steps:
        base, D = all_deltas(pi, p)
        C = {ab: w for ab, w in D.items() if w < 0}
        if not C: break
        v, mv = f(C)
        q = apply(pi, mv); qc = c_max(q, p)
        if qc >= cur:
            fallbacks += 1
            ab = min(C, key=C.get); mv = [ab]; q = apply(pi, mv); qc = c_max(q, p)
        tot_swaps += len(mv); tot_cr += crossings(mv)
        pi, cur = q, qc; steps += 1
    return {"final": cur, "steps": steps, "time_s": time.perf_counter() - t0, "fallbacks": fallbacks,
            "swaps_per_step": tot_swaps / max(steps, 1), "cr_per_step": tot_cr / max(steps, 1)}

def job_anatomy(args):
    n, m, seed = args
    p = lcg_instance(seed, n, m); rng = random.Random(seed)
    out = []
    # identity start, two random states, and states along the Motzkin descent from identity
    states = [("identity", list(range(n)))]
    for r in range(2):
        q = list(range(n)); rng.shuffle(q); states.append(("random", q))
    pi = list(range(n)); cur = c_max(pi, p); traj = []
    for step in range(200):
        base, D = all_deltas(pi, p); C = {ab: w for ab, w in D.items() if w < 0}
        if not C: break
        v, mv = select_c0(C); q = apply(pi, mv); qc = c_max(q, p)
        if qc >= cur:
            ab = min(C, key=C.get); q = apply(pi, [ab]); qc = c_max(q, p)
        pi, cur = q, qc; traj.append(pi[:])
    if traj:
        L = len(traj)
        for frac in (0.25, 0.5, 0.75):
            states.append((f"traj{int(frac*100)}", traj[min(L - 1, int(frac * L))]))
    for kind, st in states:
        rec = analyse_state(st, p, rng, with_pairs=True, with_c2=(n <= 50))
        rec.update({"n": n, "m": m, "seed": seed, "kind": kind, "traj_len": len(traj)})
        out.append(rec)
    return out

def job_descent(args):
    n, m, seed, start, variants = args
    p = lcg_instance(seed, n, m); rng = random.Random(1000 + seed + 17 * start)
    if start == 0: pi0 = list(range(n))
    else:
        pi0 = list(range(n)); rng.shuffle(pi0)
    res = {"n": n, "m": m, "seed": seed, "start": start}
    for name in variants:
        res[name] = descent(pi0[:], p, name)
    return res

if __name__ == "__main__":
    mode = sys.argv[1]
    rng = random.Random(2026)
    classes = [(20, 5), (20, 10), (20, 20), (50, 5), (50, 10), (50, 20)]
    if mode == "anatomy":
        jobs = [(n, m, rng.randrange(1, 2**31 - 1)) for (n, m) in classes for _ in range(5)]
        jobs += [(100, 10, rng.randrange(1, 2**31 - 1)) for _ in range(2)]
        with Pool(4) as pool:
            res = [r for chunk in pool.imap_unordered(job_anatomy, jobs) for r in chunk]
        json.dump(res, open(RES / "anatomy.json", "w"))
        print("states:", len(res))
    elif mode == "descent":
        jobs = []
        for (n, m) in classes:
            for _ in range(4):
                seed = rng.randrange(1, 2**31 - 1)
                for start in range(3):
                    variants = ["c0", "c1", "pa1", "inv"] + (["c2"] if n == 20 else [])
                    jobs.append((n, m, seed, start, variants))
        with Pool(4) as pool:
            res = list(pool.imap_unordered(job_descent, jobs))
        json.dump(res, open(RES / "descent.json", "w"))
        print("descents:", len(res))
