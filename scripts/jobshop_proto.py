"""Job shop prototype for the flow shop -> job shop plan (articles/jobshop/PLAN.md).

Standalone (no imports from src/), pure Python. It is a throw-away prototype for
checking the claims the plan relies on, not the production module (that is P0 in
the plan: src/jobshop/).

Model
-----
Operation id o = j*m + t (job j, t-th operation of its route), machine mach[o],
processing time dur[o]. A solution is a list of m machine sequences (operation ids).
The disjunctive graph is acyclic iff the solution is feasible; C_max is the longest
path (heads r, tails q, q excludes the operation itself).

Neighborhood construction ("virtual permutation")
-------------------------------------------------
One critical path P (list of operations) is split into blocks: maximal runs of
consecutive operations of P joined by machine arcs. A candidate is an endpoint swap
(a, b) of two operations of ONE block (block-local indices a < b), i.e. an interval
[start+a, start+b] of path positions. A compound move is a set of candidates on
pairwise disjoint intervals (k-bonacci family; k = 2 Fibonacci, k = inf Dynasearch),
non-crossing intervals (Motzkin) or arbitrary disjoint pairs (involutions). The DP
over path positions is the PFSP DP (src/neighborhoods/classical/kbonacci.py:solve_dp).

Subcommands (see main()):
  lemmas   brute-force checks of L1-L4 on random instances and states
  anatomy  per-class statistics on Taillard JSSP (and flow-shop-like) instances
  descent  best-improvement descent per neighborhood from common starts
  ils      short ILS pilot (same logic as src/algorithms/ils.py, JSSP adapter)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


class Instance:
    def __init__(self, name: str, n: int, m: int, route: list[list[int]], p: list[list[int]]):
        self.name, self.n, self.m = name, n, m
        self.route, self.p = route, p
        N = n * m
        self.N = N
        self.mach = [route[o // m][o % m] for o in range(N)]
        self.dur = [p[o // m][o % m] for o in range(N)]
        self.lb = max(
            max(sum(row) for row in p),
            max(sum(self.dur[o] for o in range(N) if self.mach[o] == k) for k in range(m)),
        )


def read_jsplib(path: str) -> Instance:
    toks = []
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("#"):
                continue
            toks.extend(line.split())
    n, m = int(toks[0]), int(toks[1])
    vals = list(map(int, toks[2 : 2 + 2 * n * m]))
    route = [[vals[2 * (j * m + t)] for t in range(m)] for j in range(n)]
    p = [[vals[2 * (j * m + t) + 1] for t in range(m)] for j in range(n)]
    return Instance(os.path.basename(path), n, m, route, p)


def lehmer(seed: int) -> int:
    a, b, c, M = 16807, 127773, 2836, 2**31 - 1
    k = seed // b
    seed = a * (seed % b) - k * c
    return seed + M if seed < 0 else seed


def unif(seed: int, low: int, high: int) -> tuple[int, int]:
    seed = lehmer(seed)
    return seed, low + int(seed / (2**31 - 1) * (high - low + 1))


def taillard_jssp(time_seed: int, machine_seed: int, n: int, m: int, rho: float = 1.0,
                  name: str = "") -> Instance:
    """Taillard (1993) JSSP generator; rho < 1 keeps each route step with prob. 1 - rho.

    rho = 1: Taillard's job shop (verified on ta01). rho = 0: every job has route
    0, 1, ..., m-1, i.e. a (general, non-permutation) flow shop F||Cmax.
    """
    p = [[0] * m for _ in range(n)]
    s = time_seed
    for j in range(n):
        for t in range(m):
            s, p[j][t] = unif(s, 1, 99)
    route = [list(range(m)) for _ in range(n)]
    s = machine_seed
    coin = random.Random(machine_seed ^ 0x5EED)
    for j in range(n):
        for t in range(m):
            s, r = unif(s, t, m - 1)
            if rho >= 1.0 or coin.random() < rho:
                route[j][t], route[j][r] = route[j][r], route[j][t]
    return Instance(name or f"gen{n}x{m}_rho{rho}", n, m, route, p)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class Eval:
    __slots__ = ("C", "r", "q", "mpred", "msucc", "pos")

    def __init__(self, C, r, q, mpred, msucc, pos):
        self.C, self.r, self.q, self.mpred, self.msucc, self.pos = C, r, q, mpred, msucc, pos


def evaluate(inst: Instance, seq: list[list[int]], need_tails: bool = True) -> Eval | None:
    """Heads, tails and C_max; None if the disjunctive graph has a cycle."""
    N, m, dur = inst.N, inst.m, inst.dur
    mpred = [-1] * N
    msucc = [-1] * N
    pos = [0] * N
    for s in seq:
        for a, o in enumerate(s):
            pos[o] = a
        for a in range(len(s) - 1):
            msucc[s[a]] = s[a + 1]
            mpred[s[a + 1]] = s[a]
    indeg = [0] * N
    for o in range(N):
        indeg[o] = (1 if o % m else 0) + (1 if mpred[o] >= 0 else 0)
    stack = [o for o in range(N) if indeg[o] == 0]
    r = [0] * N
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        fu = r[u] + dur[u]
        if (u + 1) % m:
            v = u + 1
            if r[v] < fu:
                r[v] = fu
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
        v = msucc[u]
        if v >= 0:
            if r[v] < fu:
                r[v] = fu
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) < N:
        return None
    C = max(r[o] + dur[o] for o in range(N))
    q = None
    if need_tails:
        q = [0] * N
        for u in reversed(order):
            best = 0
            if (u + 1) % m:
                v = u + 1
                best = q[v] + dur[v]
            v = msucc[u]
            if v >= 0:
                t = q[v] + dur[v]
                if t > best:
                    best = t
            q[u] = best
    return Eval(C, r, q, mpred, msucc, pos)


def cmax(inst: Instance, seq) -> int | None:
    ev = evaluate(inst, seq, need_tails=False)
    return None if ev is None else ev.C


# ---------------------------------------------------------------------------
# Critical path and blocks
# ---------------------------------------------------------------------------


def critical_path(inst: Instance, ev: Eval, prefer: str = "machine", rng: random.Random | None = None) -> list[int]:
    """One critical path, traced backwards from an operation that ends at C_max."""
    m, dur, r = inst.m, inst.dur, ev.r
    ends = [o for o in range(inst.N) if r[o] + dur[o] == ev.C]
    u = rng.choice(ends) if (rng and prefer == "random") else ends[0]
    path = [u]
    while r[u] > 0:
        jp = u - 1 if u % m else -1
        mp = ev.mpred[u]
        cand = []
        if jp >= 0 and r[jp] + dur[jp] == r[u]:
            cand.append(("job", jp))
        if mp >= 0 and r[mp] + dur[mp] == r[u]:
            cand.append(("machine", mp))
        if len(cand) == 2:
            if prefer == "random" and rng is not None:
                u = rng.choice(cand)[1]
            else:
                u = cand[0][1] if cand[0][0] == prefer else cand[1][1]
        else:
            u = cand[0][1]
        path.append(u)
    path.reverse()
    return path


def blocks_of(path: list[int], ev: Eval) -> list[tuple[int, int]]:
    """Blocks as (start index in path, length)."""
    out = []
    s = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or ev.msucc[path[t - 1]] != path[t]:
            out.append((s, t - s))
            s = t
    return out


def border_ok(a: int, b: int, L: int, bidx: int, nblocks: int) -> bool:
    """Refined block-border test (lossless filter, L3): the swap must replace the first
    or the last operation of its block; in the first block only the last counts, in the
    last block only the first counts."""
    if nblocks == 1:
        return False
    first_counts = bidx != 0
    last_counts = bidx != nblocks - 1
    return (first_counts and a == 0) or (last_counts and b == L - 1)


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


def apply_swaps(inst: Instance, seq, ev: Eval, path, swaps_path: list[tuple[int, int]]):
    """Apply endpoint swaps given as path index pairs (i, j) (same block)."""
    new = [s[:] for s in seq]
    for i, j in swaps_path:
        u, v = path[i], path[j]
        k = inst.mach[u]
        a, b = ev.pos[u], ev.pos[v]
        new[k][a], new[k][b] = new[k][b], new[k][a]
    return new


def block_candidates(path, blks, kmax: int, filt: bool = True):
    """Endpoint-swap candidates (i, j) in path indices, distance <= kmax-1, same block."""
    out = []
    nb = len(blks)
    for bidx, (s, L) in enumerate(blks):
        for a in range(L - 1):
            for b in range(a + 1, min(L - 1, a + kmax - 1) + 1):
                if filt and not border_ok(a, b, L, bidx, nb):
                    continue
                out.append((s + a, s + b))
    return out


def estimate_swap(inst: Instance, seq, ev: Eval, path, i: int, j: int) -> int:
    """Block-local estimate of C_max after swapping path[i], path[j] (same machine).

    Recomputes heads (left to right) and tails (right to left) of the operations in
    the machine interval in the NEW order, with the heads of their job predecessors /
    machine predecessor of the interval and the tails of their job successors / machine
    successor of the interval taken from the current solution (Taillard 1994 / Balas &
    Vazacopoulos 1998 style). Exact lower bound for adjacent critical swaps.
    """
    m, dur, r, q = inst.m, inst.dur, ev.r, ev.q
    u, v = path[i], path[j]
    k = inst.mach[u]
    a, b = ev.pos[u], ev.pos[v]
    seg = seq[k][a : b + 1]
    seg = [seg[-1]] + seg[1:-1] + [seg[0]]
    before = seq[k][a - 1] if a > 0 else -1
    after = seq[k][b + 1] if b + 1 < len(seq[k]) else -1
    hr = []
    prev_end = (r[before] + dur[before]) if before >= 0 else 0
    for o in seg:
        jp = o - 1 if o % m else -1
        h = max(prev_end, (r[jp] + dur[jp]) if jp >= 0 else 0)
        hr.append(h)
        prev_end = h + dur[o]
    tq = [0] * len(seg)
    nxt = (q[after] + dur[after]) if after >= 0 else 0
    for t in range(len(seg) - 1, -1, -1):
        o = seg[t]
        js = o + 1 if (o + 1) % m else -1
        tail = max(nxt, (q[js] + dur[js]) if js >= 0 else 0)
        tq[t] = tail
        nxt = tail + dur[o]
    return max(hr[t] + dur[seg[t]] + tq[t] for t in range(len(seg)))


def swap_feasible(inst: Instance, ev: Eval, path, i: int, j: int) -> bool:
    """Endpoint swap (path[i], path[j]) inside one block: feasible iff no reversed pair
    (x, y) has an E-path x -> y (block-reordering theorem). Reversed pairs of an endpoint
    swap: (path[i], path[y]) for y in (i, j] and (path[x], path[j]) for x in [i, j).
    Heads certificate first (r grows strictly along any path), DFS only if needed."""
    if j == i + 1:
        return True  # adjacent critical swap (van Laarhoven et al. 1992)
    m, r = inst.m, ev.r
    U, V = path[i], path[j]
    jsU = U + 1 if (U + 1) % m else -1
    jpV = V - 1 if V % m else -1
    if jsU >= 0:
        for y in path[i + 1 : j + 1]:
            jp = y - 1 if y % m else -1
            if jp >= 0 and (jp == jsU or r[jp] > r[jsU]) and reach(inst, ev, jsU, jp):
                return False
    if jpV >= 0:
        for x in path[i:j]:
            js = x + 1 if (x + 1) % m else -1
            if js >= 0 and (js == jpV or r[jpV] > r[js]) and reach(inst, ev, js, jpV):
                return False
    return True


def solve_dp(n: int, candidates):
    """PFSP DP over positions (copied from src/neighborhoods/classical/kbonacci.py)."""
    ending_at: dict[int, list] = {}
    for c in candidates:
        ending_at.setdefault(c[1], []).append(c)
    best = [0] * (n + 1)
    choice = [None] * (n + 1)
    for p in range(1, n + 1):
        best[p] = best[p - 1]
        for c in ending_at.get(p - 1, ()):
            val = best[c[0]] + c[2]
            if val < best[p]:
                best[p] = val
                choice[p] = c
    chosen = []
    p = n
    while p > 0:
        c = choice[p]
        if c is None:
            p -= 1
        else:
            chosen.append((c[0], c[1]))
            p = c[0]
    chosen.reverse()
    return best[n], chosen


def count_compound(L: int, family: str, k: int = 2) -> int:
    """Number of compound moves (incl. the empty one) inside one block of length L."""
    if family in ("fib", "kbonacci", "dyn"):
        kk = {"fib": 2, "dyn": max(L, 2)}.get(family, k)
        a = [1] + [0] * L
        for p in range(1, L + 1):
            a[p] = sum(a[p - t] for t in range(1, min(kk, p) + 1))
        return a[L]
    if family == "motzkin":
        M = [1, 1]
        for t in range(2, L + 1):
            M.append(M[t - 1] + sum(M[s] * M[t - 2 - s] for s in range(t - 1)))
        return M[L]
    if family == "inv":
        I = [1, 1]
        for t in range(2, L + 1):
            I.append(I[t - 1] + (t - 1) * I[t - 2])
        return I[L]
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Starting solutions and perturbation
# ---------------------------------------------------------------------------


def decode_opseq(inst: Instance, opseq: list[int]) -> list[list[int]]:
    """Operation-based sequence with repetition (job ids) -> machine sequences
    (semi-active, always feasible)."""
    m = inst.m
    nxt = [0] * inst.n
    seq = [[] for _ in range(m)]
    for j in opseq:
        o = j * m + nxt[j]
        nxt[j] += 1
        seq[inst.mach[o]].append(o)
    return seq


def identity_start(inst: Instance) -> list[list[int]]:
    """JSSP analog of the identity permutation: jobs dispatched round-robin by index."""
    return decode_opseq(inst, [j for _ in range(inst.m) for j in range(inst.n)])


def random_start(inst: Instance, rng: random.Random) -> list[list[int]]:
    ops = [j for j in range(inst.n) for _ in range(inst.m)]
    rng.shuffle(ops)
    return decode_opseq(inst, ops)


def topo_opseq(inst: Instance, seq) -> list[int]:
    """Machine sequences -> operation-based sequence (a topological order, job ids)."""
    ev = evaluate(inst, seq, need_tails=False)
    order = sorted(range(inst.N), key=lambda o: (ev.r[o], o))
    return [o // inst.m for o in order]


def double_bridge_jssp(inst: Instance, seq, rng: random.Random) -> list[list[int]]:
    """Double bridge on the operation-based sequence + semi-active decode (always feasible)."""
    ops = topo_opseq(inst, seq)
    n = len(ops)
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return decode_opseq(inst, ops[:a] + ops[b:c] + ops[a:b] + ops[c:])


# ---------------------------------------------------------------------------
# Neighborhood search step
# ---------------------------------------------------------------------------


def best_move(inst: Instance, seq, ev: Eval, mode: str, exact: bool = True, stats: dict | None = None):
    """Best compound move of a family on one critical path.

    mode: 'n1' (best single adjacent critical swap), 'n5' (Nowicki-Smutnicki: first/last
    pair of each block), 'fib', 'k3', 'k4', 'k6', 'k8', 'dyn' (k = inf); exact=True uses
    exact single-swap deltas (one evaluation each), else the block-local estimate.
    Returns (new_seq, new_C, swaps) or (None, None, []) when there is no candidate.
    """
    path = critical_path(inst, ev)
    blks = blocks_of(path, ev)
    if mode in ("n1", "n5"):
        kmax = 2
    elif mode == "fib":
        kmax = 2
    elif mode == "dyn":
        kmax = 10**9
    else:
        kmax = int(mode[1:])
    cands = block_candidates(path, blks, kmax, filt=(mode != "n1"))  # N1 = all adjacent critical pairs
    if mode == "n5":
        nb = len(blks)
        keep = []
        for bidx, (s, L) in enumerate(blks):
            if L < 2:
                continue
            if bidx != 0:
                keep.append((s, s + 1))
            if bidx != nb - 1:
                keep.append((s + L - 2, s + L - 1))
        cands = sorted(set(keep))
    scored = []
    n_infeasible = 0
    for i, j in cands:
        if exact:
            c = cmax(inst, apply_swaps(inst, seq, ev, path, [(i, j)]))
            if c is None:
                n_infeasible += 1
                continue
        else:
            if not swap_feasible(inst, ev, path, i, j):
                n_infeasible += 1
                continue
            c = estimate_swap(inst, seq, ev, path, i, j)
        scored.append((i, j, c - ev.C))
    if not scored:
        return None, None, []
    if mode in ("n1", "n5"):
        i, j, d = min(scored, key=lambda x: x[2])
        chosen = [(i, j)]
        pred = d
    else:
        pred, chosen = solve_dp(len(path), scored)
        if not chosen:
            i, j, pred = min(scored, key=lambda x: x[2])
            chosen = [(i, j)]
    new = apply_swaps(inst, seq, ev, path, chosen)
    newC = cmax(inst, new)
    if stats is not None:
        stats.update(dict(cands=len(cands), infeasible=n_infeasible, pred=pred,
                          real=(newC - ev.C) if newC is not None else None, nswaps=len(chosen),
                          dist=[j - i for i, j in chosen], path_len=len(path), nblocks=len(blks)))
    if newC is None:  # cannot happen for disjoint intervals (L2), kept as a guard
        return None, None, []
    move_id = frozenset((path[i], path[j]) for i, j in chosen)
    return new, newC, move_id


# ---------------------------------------------------------------------------
# Subcommand: lemmas
# ---------------------------------------------------------------------------


def random_instance(rng: random.Random, n: int, m: int, rho: float = 1.0) -> Instance:
    return taillard_jssp(rng.randrange(1, 2**31 - 1), rng.randrange(1, 2**31 - 1), n, m, rho)


def states_for(inst: Instance, rng: random.Random, nrand: int, nlocal: int):
    out = []
    for _ in range(nrand):
        out.append(("random", random_start(inst, rng)))
    for _ in range(nlocal):
        seq = random_start(inst, rng)
        for _ in range(200):
            ev = evaluate(inst, seq)
            new, newC, _ = best_move(inst, seq, ev, "n1")
            if new is None or newC >= ev.C:
                break
            seq = new
        out.append(("local", seq))
    return out


def cmd_lemmas(args) -> dict:
    rng = random.Random(args.seed)
    res = defaultdict(int)
    examples = {}
    sizes = [(4, 3), (5, 3), (6, 4), (8, 4), (10, 5), (10, 10), (15, 5), (20, 5)]
    t0 = time.time()
    for rep in range(args.reps):
        n, m = sizes[rep % len(sizes)]
        rho = [1.0, 1.0, 0.5, 0.0][rep % 4]
        inst = random_instance(rng, n, m, rho)
        for kind, seq in states_for(inst, rng, 2, 2):
            ev = evaluate(inst, seq)
            for prefer in ("machine", "job"):
                path = critical_path(inst, ev, prefer)
                blks = blocks_of(path, ev)
                nb = len(blks)
                # L1 + L3 + feasibility by distance, for every swap inside every block
                single = {}
                for bidx, (s, L) in enumerate(blks):
                    for a in range(L - 1):
                        for b in range(a + 1, L):
                            new = apply_swaps(inst, seq, ev, path, [(s + a, s + b)])
                            c = cmax(inst, new)
                            d = b - a
                            res[f"swap_total_d{min(d, 6)}"] += 1
                            if c is None:
                                res[f"swap_infeasible_d{min(d, 6)}"] += 1
                                if d == 1:
                                    res["L1_violations"] += 1
                                continue
                            single[(s + a, s + b)] = c - ev.C
                            if c < ev.C:
                                res["improving_swaps"] += 1
                                if not border_ok(a, b, L, bidx, nb):
                                    res["L3_violations"] += 1
                                    examples.setdefault("L3", (inst.name, kind, prefer, a, b, L, bidx, nb))
                            if (estimate_swap(inst, seq, ev, path, s + a, s + b) if d == 1 else None) is not None:
                                est = estimate_swap(inst, seq, ev, path, s + a, s + b)
                                res["est_adj_total"] += 1
                                if est > c:
                                    res["est_adj_above_real"] += 1  # must stay 0: lower bound
                                if (est >= ev.C) and est != c:
                                    res["est_adj_nonimproving_not_exact"] += 1
                # L2: random sets of individually feasible swaps on disjoint intervals
                feas = list(single.keys())
                for _ in range(30):
                    if len(feas) < 2:
                        break
                    rng.shuffle(feas)
                    chosen = []
                    for (i, j) in feas:
                        if all(j < i2 or j2 < i for i2, j2 in chosen):
                            if rng.random() < 0.7:
                                chosen.append((i, j))
                    if len(chosen) < 2:
                        continue
                    res["L2_compound_tested"] += 1
                    if cmax(inst, apply_swaps(inst, seq, ev, path, chosen)) is None:
                        res["L2_violations"] += 1
                        examples.setdefault("L2", (inst.name, chosen))
                # L4: nested / crossing pairs of individually feasible swaps in one block
                for (i1, j1), (i2, j2) in itertools.combinations(feas, 2):
                    same_block = any(s <= i1 and j1 < s + L and s <= i2 and j2 < s + L for s, L in blks)
                    if not same_block:
                        continue
                    nested = (i1 < i2 and j2 < j1) or (i2 < i1 and j1 < j2)
                    crossing = (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1)
                    if not (nested or crossing):
                        continue
                    key = "nested" if nested else "crossing"
                    res[f"L4_{key}_pairs"] += 1
                    if cmax(inst, apply_swaps(inst, seq, ev, path, [(i1, j1), (i2, j2)])) is None:
                        res[f"L4_{key}_infeasible"] += 1
                        if key == "nested":
                            examples.setdefault("L4_nested", json.dumps(
                                {"route": inst.route, "p": inst.p, "seq": seq, "path": path,
                                 "swaps": [(i1, j1), (i2, j2)]}))
                # L4m / L4i: random Motzkin (non-crossing, nesting allowed) and involution
                # (crossing allowed) sets of >= 2 individually feasible swaps
                for fam in ("motzkin", "inv"):
                    for _ in range(20):
                        if len(feas) < 2:
                            break
                        rng.shuffle(feas)
                        chosen = []
                        for (i, j) in feas:
                            ends = {x for c in chosen for x in c}
                            if i in ends or j in ends:
                                continue
                            if fam == "motzkin" and any(i < i2 < j < j2 or i2 < i < j2 < j for i2, j2 in chosen):
                                continue
                            if rng.random() < 0.7:
                                chosen.append((i, j))
                        if len(chosen) < 2:
                            continue
                        nested = any((i1 < i2 and j2 < j1) for (i1, j1) in chosen for (i2, j2) in chosen)
                        tag = f"L4{fam[0]}_{'with_nesting' if nested else 'disjoint_only'}"
                        if fam == "inv":
                            crossing = any(i1 < i2 < j1 < j2 for (i1, j1) in chosen for (i2, j2) in chosen)
                            tag = f"L4i_{'with_crossing' if crossing else 'no_crossing'}"
                        res[tag + "_tested"] += 1
                        if cmax(inst, apply_swaps(inst, seq, ev, path, chosen)) is None:
                            res[tag + "_infeasible"] += 1
                            examples.setdefault(tag, (inst.name, n, m, rho, kind, prefer, chosen))
                # L5 (DP invariance): filtered vs unfiltered DP optimum on exact deltas
                cand_all = [(i, j, d) for (i, j), d in single.items()]
                cand_f = []
                for (i, j), d in single.items():
                    for bidx, (s, L) in enumerate(blks):
                        if s <= i and j < s + L and border_ok(i - s, j - s, L, bidx, nb):
                            cand_f.append((i, j, d))
                if cand_all:
                    v_all, _ = solve_dp(len(path), cand_all)
                    v_f, _ = solve_dp(len(path), cand_f) if cand_f else (0, [])
                    res["L5_dp_tested"] += 1
                    if v_all != v_f:
                        res["L5_violations"] += 1
    res["seconds"] = round(time.time() - t0, 1)
    out = {"counts": dict(res), "examples": {k: str(v) for k, v in examples.items()}}
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# Subcommand: anatomy
# ---------------------------------------------------------------------------

TA_CLASSES = {
    "15x15": range(1, 11), "20x15": range(11, 21), "20x20": range(21, 31), "30x15": range(31, 41),
    "30x20": range(41, 51), "50x15": range(51, 61), "50x20": range(61, 71), "100x20": range(71, 81),
}


def load_ta(dirpath: str, idx: int) -> Instance:
    return read_jsplib(os.path.join(dirpath, f"ta{idx:02d}"))


def local_opt(inst: Instance, seq, mode: str = "n5", max_steps: int = 10**6):
    steps = 0
    while steps < max_steps:
        ev = evaluate(inst, seq)
        new, newC, _ = best_move(inst, seq, ev, mode)
        if new is None or newC >= ev.C:
            return seq, ev.C, steps
        seq = new
        steps += 1
    return seq, cmax(inst, seq), steps


def anatomy_state(inst: Instance, seq, ks=(2, 3, 4, 6, 8, 10**9), check_feas=True):
    ev = evaluate(inst, seq)
    path = critical_path(inst, ev)
    blks = blocks_of(path, ev)
    Ls = [L for _, L in blks]
    row = {"C": ev.C, "path_len": len(path), "nblocks": len(blks),
           "block_len_mean": statistics.mean(Ls), "block_len_max": max(Ls),
           "blocks_ge2": sum(1 for L in Ls if L >= 2), "ops_in_blocks_ge2": sum(L for L in Ls if L >= 2)}
    # log10 neighborhood sizes (product over blocks), unfiltered
    for fam in ("fib", "k3", "k4", "dyn", "motzkin", "inv"):
        if fam.startswith("k"):
            v = sum(math.log10(count_compound(L, "kbonacci", int(fam[1:]))) for L in Ls)
        else:
            v = sum(math.log10(count_compound(L, fam)) for L in Ls)
        row[f"log10N_{fam}"] = round(v, 3)
    for k in ks:
        kk = "inf" if k >= 10**9 else str(k)
        cands = block_candidates(path, blks, k, filt=True)
        row[f"cands_k{kk}"] = len(cands)
        if check_feas:
            inf = 0
            improving = 0
            for i, j in cands:
                c = cmax(inst, apply_swaps(inst, seq, ev, path, [(i, j)]))
                if c is None:
                    inf += 1
                elif c < ev.C:
                    improving += 1
            row[f"infeasible_k{kk}"] = inf
            row[f"improving_k{kk}"] = improving
    return row


def cmd_anatomy(args) -> dict:
    rng = random.Random(args.seed)
    out = {}
    classes = args.classes.split(",") if args.classes else list(TA_CLASSES)
    for cls in classes:
        rows = []
        for idx in list(TA_CLASSES[cls])[: args.instances]:
            inst = load_ta(args.jsplib, idx)
            for s in range(args.states):
                seq = random_start(inst, rng) if s % 2 == 0 else identity_start(inst)
                seq, C, steps = local_opt(inst, seq, "n5")
                rows.append(anatomy_state(inst, seq, check_feas=not args.no_feas))
        agg = {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]}
        agg["block_len_q90"] = 0
        out[cls] = agg
        print(cls, json.dumps(agg))
        sys.stdout.flush()
    return out


def cmd_continuum(args) -> dict:
    """Flow shop (rho=0) -> job shop (rho=1) on Taillard-like generated instances."""
    out = {}
    for n, m in [(20, 5), (20, 10), (30, 15), (50, 10)]:
        for rho in (0.0, 0.1, 0.25, 0.5, 1.0):
            rng = random.Random(args.seed + n * 1000 + m)
            rows = []
            for t in range(args.instances):
                inst = taillard_jssp(rng.randrange(1, 2**31 - 1), rng.randrange(1, 2**31 - 1), n, m, rho)
                for s in range(args.states):
                    seq = random_start(inst, rng) if s % 2 == 0 else identity_start(inst)
                    seq, C, _ = local_opt(inst, seq, "n5")
                    row = anatomy_state(inst, seq, ks=(2, 3, 4, 8, 10**9), check_feas=True)
                    row["rpd_lb"] = 100.0 * (C - inst.lb) / inst.lb
                    rows.append(row)
            agg = {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]}
            out[f"{n}x{m}_rho{rho}"] = agg
            print(f"{n}x{m} rho={rho}", json.dumps(agg))
            sys.stdout.flush()
    return out


# ---------------------------------------------------------------------------
# Subcommand: descent and ILS pilots
# ---------------------------------------------------------------------------


def cmd_descent(args) -> dict:
    modes = args.modes.split(",")
    out = defaultdict(list)
    rng = random.Random(args.seed)
    for name in args.files.split(","):
        inst = read_jsplib(os.path.join(args.jsplib, name))
        bks = BKS.get(name, inst.lb)
        starts = [identity_start(inst)] + [random_start(inst, rng) for _ in range(args.starts - 1)]
        for si, s0 in enumerate(starts):
            for mode in modes:
                t0 = time.time()
                seq, C, steps = local_opt(inst, s0, mode)
                out[mode].append({"inst": name, "start": si, "rpd": 100.0 * (C - bks) / bks,
                                  "steps": steps, "sec": time.time() - t0})
        print(name, {mo: round(statistics.mean(r["rpd"] for r in out[mo] if r["inst"] == name), 2)
                     for mo in modes})
        sys.stdout.flush()
    summ = {mo: {"rpd": round(statistics.mean(r["rpd"] for r in v), 3),
                 "steps": round(statistics.mean(r["steps"] for r in v), 1),
                 "sec": round(statistics.mean(r["sec"] for r in v), 3)} for mo, v in out.items()}
    print(json.dumps(summ, indent=1))
    return {"summary": summ, "runs": dict(out)}


def ils_jssp(inst: Instance, mode: str, time_ms: int, seed: int, tabu_tenure: int = 10, elite_k: int = 10,
             exact: bool = True):
    """Same logic as src/algorithms/ils.py (tabu on move id + aspiration, elite-pool
    double-bridge kick), with the JSSP adapter."""
    rng = random.Random(seed)
    seq = identity_start(inst)
    C = cmax(inst, seq)
    best_seq, best_C = seq, C
    elite = [(C, seq)]
    cursor = 0
    tabu: dict = {}
    it = 0
    t0 = time.time()
    while (time.time() - t0) * 1000 < time_ms:
        ev = evaluate(inst, seq)
        new, newC, mid = best_move(inst, seq, ev, mode, exact=exact)
        tabu_active = new is not None and tabu.get(mid, -1) > it
        if new is None or (tabu_active and newC >= best_C):
            src = elite[cursor % len(elite)][1]
            cursor += 1
            new = double_bridge_jssp(inst, src, rng)
            newC = cmax(inst, new)
            mid = ("kick", it)
        seq, C = new, newC
        tabu[mid] = it + tabu_tenure
        if C < best_C:
            best_C, best_seq = C, seq
            if all(C != e[0] for e in elite):
                elite.append((C, seq))
                elite.sort(key=lambda x: x[0])
                elite = elite[:elite_k]
                cursor = 0
        it += 1
    return best_C, it


def cmd_ils(args) -> dict:
    modes = args.modes.split(",")
    res = defaultdict(list)
    for name in args.files.split(","):
        inst = read_jsplib(os.path.join(args.jsplib, name))
        bks = BKS.get(name, inst.lb)
        for mode in modes:
            for seed in range(args.seeds):
                C, it = ils_jssp(inst, mode, args.time_ms, seed, exact=not args.estimate)
                res[mode].append({"inst": name, "seed": seed, "rpd": 100.0 * (C - bks) / bks, "iters": it})
        print(name, {mo: round(statistics.mean(r["rpd"] for r in res[mo] if r["inst"] == name), 2)
                     for mo in modes},
              {mo: round(statistics.mean(r["iters"] for r in res[mo] if r["inst"] == name)) for mo in modes})
        sys.stdout.flush()
    summ = {mo: {"rpd": round(statistics.mean(r["rpd"] for r in v), 3),
                 "iters": round(statistics.mean(r["iters"] for r in v), 1)} for mo, v in res.items()}
    print(json.dumps(summ, indent=1))
    return {"summary": summ, "runs": dict(res)}


# Optima / best known upper bounds used only for the pilot (JSPLIB, ft/la/ta optima).
BKS = {"ft06": 55, "ft10": 930, "ft20": 1165, "la16": 945, "la17": 784, "la18": 848, "la19": 842,
       "la20": 902, "ta01": 1231, "ta02": 1244, "ta03": 1218, "ta04": 1175, "ta05": 1224,
       "abz5": 1234, "abz6": 943, "orb01": 1059, "orb02": 888}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lemmas", "anatomy", "continuum", "descent", "ils", "theorem", "additivity"])
    ap.add_argument("--jsplib", default="data/JSPLIB/instances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--classes", default="")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--states", type=int, default=4)
    ap.add_argument("--no-feas", action="store_true")
    ap.add_argument("--files", default="ft10,la16,la17,ta01,ta02")
    ap.add_argument("--modes", default="n1,n5,fib,k3,k4,dyn")
    ap.add_argument("--starts", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--time-ms", type=int, default=1000)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.setrecursionlimit(10000)
    fn = {"lemmas": cmd_lemmas, "anatomy": cmd_anatomy, "continuum": cmd_continuum,
          "descent": cmd_descent, "ils": cmd_ils, "theorem": globals()["cmd_theorem"],
          "additivity": globals()["cmd_additivity"]}[args.cmd]
    out = fn(args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)



# ---------------------------------------------------------------------------
# Block-reordering feasibility theorem: checks (subcommand "theorem")
# ---------------------------------------------------------------------------


def reach(inst: Instance, ev: Eval, src: int, dst: int) -> bool:
    """Is dst reachable from src in the CURRENT graph (job + machine arcs)? Head-pruned DFS."""
    if src == dst:
        return True
    m, r = inst.m, ev.r
    lim = r[dst]
    stack, seen = [src], {src}
    while stack:
        u = stack.pop()
        for v in ((u + 1) if (u + 1) % m else -1, ev.msucc[u]):
            if v < 0 or v in seen or r[v] > lim:
                continue
            if v == dst:
                return True
            seen.add(v)
            stack.append(v)
    return False


def e_path(inst: Instance, ev: Eval, x: int, y: int) -> bool:
    """Path x -> y that leaves x by its job arc and enters y by its job arc (x, y in one block)."""
    m = inst.m
    if (x + 1) % m == 0 or y % m == 0:
        return False
    return reach(inst, ev, x + 1, y - 1)


def predicted_infeasible(inst, ev, path, blks, new_orders) -> bool:
    """Theorem: infeasible iff some block has a reversed pair (x, y) with an E-path x -> y."""
    for (s, L), new in zip(blks, new_orders):
        old = path[s : s + L]
        if new == old:
            continue
        npos = {o: t for t, o in enumerate(new)}
        for a in range(L):
            for b in range(a + 1, L):
                x, y = old[a], old[b]
                if npos[y] < npos[x] and e_path(inst, ev, x, y):
                    return True
    return False


def apply_block_orders(inst, seq, ev, path, blks, new_orders):
    new = [s_[:] for s_ in seq]
    for (s, L), order in zip(blks, new_orders):
        k = inst.mach[path[s]]
        p0 = ev.pos[path[s]]
        new[k][p0 : p0 + L] = order
    return new


def cmd_theorem(args) -> dict:
    rng = random.Random(args.seed)
    res = defaultdict(int)
    sizes = [(5, 3), (6, 4), (8, 4), (10, 5), (10, 10), (15, 5), (20, 5), (15, 15)]
    for rep in range(args.reps):
        n, m = sizes[rep % len(sizes)]
        rho = [1.0, 1.0, 0.5, 0.0][rep % 4]
        inst = random_instance(rng, n, m, rho)
        for kind, seq in states_for(inst, rng, 2, 2):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev, "machine" if rep % 2 else "job")
            blks = blocks_of(path, ev)
            big = [b for b, (s, L) in enumerate(blks) if L >= 3]
            if not big:
                continue
            for trial in range(25):
                # (a) arbitrary permutations of random segments in 1..3 random blocks
                orders = [path[s : s + L] for s, L in blks]
                for b in rng.sample(big, min(len(big), rng.choice([1, 1, 2, 3]))):
                    s, L = blks[b]
                    lo = rng.randrange(0, L - 1)
                    hi = rng.randrange(lo + 1, L)
                    seg = orders[b][lo : hi + 1]
                    rng.shuffle(seg)
                    orders[b] = orders[b][:lo] + seg + orders[b][hi + 1 :]
                real_inf = cmax(inst, apply_block_orders(inst, seq, ev, path, blks, orders)) is None
                pred_inf = predicted_infeasible(inst, ev, path, blks, orders)
                res["perm_tested"] += 1
                res["perm_infeasible"] += real_inf
                if real_inf != pred_inf:
                    res["perm_mismatch"] += 1
            # (b) pairwise property for involutions (disjoint transpositions, crossing allowed)
            swaps = [(s + a, s + b) for s, L in blks for a in range(L) for b in range(a + 1, L)]
            single_ok = {sw: cmax(inst, apply_swaps(inst, seq, ev, path, [sw])) is not None for sw in swaps}
            pair_cache: dict = {}

            def pair_ok(s1, s2):
                key = (min(s1, s2), max(s1, s2))
                if key not in pair_cache:
                    pair_cache[key] = cmax(inst, apply_swaps(inst, seq, ev, path, list(key))) is not None
                return pair_cache[key]

            for trial in range(25):
                rng.shuffle(swaps)
                chosen = []
                for (i, j) in swaps:
                    if any(i in c or j in c for c in chosen):
                        continue
                    if rng.random() < 0.5:
                        chosen.append((i, j))
                if len(chosen) < 3:
                    continue
                new_seq = apply_swaps(inst, seq, ev, path, chosen)
                real_inf = cmax(inst, new_seq) is None
                orders = []
                for s, L in blks:
                    k = inst.mach[path[s]]
                    p0 = ev.pos[path[s]]
                    orders.append(new_seq[k][p0 : p0 + L])
                theorem_inf = predicted_infeasible(inst, ev, path, blks, orders)
                naive_inf = (not all(single_ok[c] for c in chosen)) or any(
                    not pair_ok(c1, c2) for c1, c2 in itertools.combinations(chosen, 2))
                crossing = any(i1 < i2 < j1 < j2 for (i1, j1) in chosen for (i2, j2) in chosen)
                res["inv_tested"] += 1
                res["inv_infeasible"] += real_inf
                res["inv_with_crossing"] += crossing
                if real_inf != theorem_inf:
                    res["inv_theorem_mismatch"] += 1
                if real_inf != naive_inf:
                    res["inv_naive_pairwise_mismatch"] += 1
                    res["inv_naive_mismatch_rescue" if naive_inf else "inv_naive_mismatch_other"] += 1
                    if not crossing:
                        res["inv_naive_mismatch_without_crossing"] += 1
            # (c) cheap heads test (sufficient for feasibility) for single endpoint swaps
            for s, L in blks:
                for a in range(L - 1):
                    for b in range(a + 2, L):
                        i, j = s + a, s + b
                        U, V = path[i], path[j]
                        ok = cmax(inst, apply_swaps(inst, seq, ev, path, [(i, j)])) is not None
                        jsU = U + 1 if (U + 1) % inst.m else -1
                        jpV = V - 1 if V % inst.m else -1
                        cert = True
                        for y in path[i + 1 : j + 1]:
                            jp = y - 1 if y % inst.m else -1
                            if jsU >= 0 and jp >= 0 and (jp == jsU or ev.r[jp] > ev.r[jsU]):
                                cert = False
                        for x in path[i : j]:
                            js = x + 1 if (x + 1) % inst.m else -1
                            if jpV >= 0 and js >= 0 and (js == jpV or ev.r[jpV] > ev.r[js]):
                                cert = False
                        res["heads_test_swaps"] += 1
                        res["heads_test_feasible"] += ok
                        res["heads_test_certified"] += cert
                        if cert and not ok:
                            res["heads_test_wrong"] += 1
    out = dict(res)
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# Subcommand: additivity of compound moves (predicted sum of single deltas vs real)
# ---------------------------------------------------------------------------


def cmd_additivity(args) -> dict:
    rng = random.Random(args.seed)
    fams = ["fib", "k3", "k4", "k8", "dyn"]
    rows = defaultdict(list)
    for name in args.files.split(","):
        inst = read_jsplib(os.path.join(args.jsplib, name))
        for st in range(args.states):
            seq = random_start(inst, rng)
            # walk a few N5 steps so that states range from random to near-local-optimal
            for _ in range(rng.randrange(0, 12)):
                ev = evaluate(inst, seq)
                new, newC, _ = best_move(inst, seq, ev, "n5")
                if new is None or newC >= ev.C:
                    break
                seq = new
            ev = evaluate(inst, seq)
            for fam in fams:
                for exact in (True, False):
                    stats: dict = {}
                    new, newC, _ = best_move(inst, seq, ev, fam, exact=exact, stats=stats)
                    if new is None or stats.get("nswaps", 0) < 2 or stats["pred"] >= 0:
                        continue
                    rows[(fam, exact)].append((stats["pred"], stats["real"], stats["nswaps"]))
    out = {}
    for (fam, exact), v in rows.items():
        n = len(v)
        out[f"{fam}_{'exact' if exact else 'estimate'}"] = {
            "moves": n,
            "mean_swaps": round(statistics.mean(x[2] for x in v), 2),
            "exact_hit_pct": round(100 * sum(1 for p_, r_, _ in v if p_ == r_) / n, 1),
            "real_worse_than_pred_pct": round(100 * sum(1 for p_, r_, _ in v if r_ > p_) / n, 1),
            "real_better_than_pred_pct": round(100 * sum(1 for p_, r_, _ in v if r_ < p_) / n, 1),
            "realized_gain_pct": round(100 * sum(r_ for _, r_, _ in v) / sum(p_ for p_, _, _ in v), 1),
            "real_nonimproving_pct": round(100 * sum(1 for _, r_, _ in v if r_ >= 0) / n, 1),
        }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
