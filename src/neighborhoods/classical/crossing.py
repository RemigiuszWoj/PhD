"""Overlap axis of compound endpoint-swap neighborhoods: Dynasearch -> Motzkin -> crossings -> involutions.

A compound move is a set of endpoint swaps (i, j), i < j, with pairwise distinct endpoints, i.e. an
involution sigma; the neighbor is pi o sigma (jobs strictly inside an interval stay in place).
Families, ordered by how much two swaps may overlap (articles/crossings/PLAN.md, section 2):

    dyn  pairwise disjoint intervals           Dynasearch, identical to k-bonacci with k = n
    c0   no crossing pair, nesting allowed     Motzkin
    c1   at most one crossing pair in the move N_1
    pa1  every arc crossed at most once        P_1 ("single crossings" per arc)
    inv  any involution                        maximum-weight matching

Mode names:  cross_<family>[_safe|_prefix][_cp]   and   cross_swap
    plain    the selected move is applied as is (semantics of the old Motzkin and of k-bonacci)
    _safe    the better (by real C_max) of the selected move and the best single swap
    _prefix  the best of the best single swap and every prefix of the move sorted by delta
    _cp      exact critical-path filter instead of the last-machine test (ablation, as in k-bonacci)
    cross_swap  the best single endpoint swap (control)

Candidates and deltas come from kbonacci.compute_candidates with k = n, so every family sees exactly
the deltas and the block filter of k-bonacci and of the previous papers. Selection maximizes the sum
of individual deltas over improving candidates only (the families are hereditary, so an arc with
delta >= 0 never improves the predicted optimum); if no candidate improves, the best single swap is
applied, like force_move_if_none in k-bonacci and Motzkin.

On dense states (many improving candidates, e.g. right after the start or after a bad move) the
selection is restricted to the TOP_K most improving candidates, and the H-cores of c1 / pa1 to the
CORE_K most improving ones; per-run counters record how often a cap was binding.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from src.neighborhoods.classical.kbonacci import Candidate, compute_candidates, solve_dp
from src.permutation_procesing import c_max

Arc = Tuple[int, int]
FAMILIES = ("dyn", "c0", "c1", "pa1", "inv", "swap")
RULES = ("plain", "safe", "prefix")
TOP_K = 3000   # cap on the candidates used by c0 / c1 / pa1 / inv on dense states
CORE_K = 400   # cap on the candidates that may form an H-core (c1, pa1)
INF = float("inf")


# ---------------------------------------------------------------------------
# Per-run statistics (reset by the experiment runner before every run)
# ---------------------------------------------------------------------------
_STAT_KEYS = ("moves", "swaps", "crossings", "nestings", "fallback", "no_improving",
              "cap_top", "cap_core", "candidates", "improving", "pred_sum", "real_sum",
              "select_ms", "total_ms")
_stats: Dict[str, float] = {k: 0 for k in _STAT_KEYS}


def reset_crossing_stats() -> None:
    for k in _STAT_KEYS:
        _stats[k] = 0


def get_crossing_stats() -> Dict[str, float]:
    return dict(_stats)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def crossings(arcs: List[Arc]) -> int:
    """Number of crossing pairs (a < c < b < d) in a set of arcs."""
    arcs = sorted(arcs)
    cr = 0
    for x in range(len(arcs)):
        _a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if c < b < d:
                cr += 1
    return cr


def nestings(arcs: List[Arc]) -> int:
    """Number of nested pairs (a < c < d < b)."""
    arcs = sorted(arcs)
    ne = 0
    for x in range(len(arcs)):
        _a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            _c, d = arcs[y]
            if d < b:
                ne += 1
    return ne


def max_crossings_per_arc(arcs: List[Arc]) -> int:
    arcs = sorted(arcs)
    cnt = [0] * len(arcs)
    for x in range(len(arcs)):
        _a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if c < b < d:
                cnt[x] += 1
                cnt[y] += 1
    return max(cnt) if cnt else 0


# ---------------------------------------------------------------------------
# Counting (PLAN.md, T1 and T4)
# ---------------------------------------------------------------------------
def count_by_crossings(n: int, c_max_: int) -> List[int]:
    """[#involutions of [n] with exactly c crossings for c = 0..c_max_] (Flajolet J-fraction:
    Motzkin paths, a down step from height h carries 0..h-1 crossings)."""
    C = c_max_
    cur = {0: [1] + [0] * C}
    for t in range(1, n + 1):
        nxt: Dict[int, List[int]] = {}
        for h, poly in cur.items():
            for nh in (h, h + 1):                 # level step (fixed point), up step (open an arc)
                acc = nxt.setdefault(nh, [0] * (C + 1))
                for i in range(C + 1):
                    acc[i] += poly[i]
            if h > 0:                             # down step: close one of h open arcs
                acc = nxt.setdefault(h - 1, [0] * (C + 1))
                for i in range(C + 1):
                    if poly[i]:
                        for j in range(min(h, C + 1 - i)):
                            acc[i + j] += poly[i]
        cur = {h: p for h, p in nxt.items() if h <= n - t}
    return cur.get(0, [0] * (C + 1))


def count_moves(n: int, family: str) -> int:
    """|N(n)| for a family (the identity move included)."""
    if family == "dyn":
        return 2 ** (n - 1) if n >= 1 else 1
    if family == "c0":
        return count_by_crossings(n, 0)[0]
    if family == "c1":
        return sum(count_by_crossings(n, 1))
    if family == "inv":
        a, b = 1, 1
        for k in range(2, n + 1):
            a, b = b, b + (k - 1) * a
        return b if n >= 1 else 1
    if family == "pa1":                           # S = 1 + xS + x^2 S^2 + x^4 S^4
        S = [1] + [0] * n
        for _ in range(n + 1):
            S2 = _poly_mul(S, S, n)
            S4 = _poly_mul(S2, S2, n)
            new = [1] + [S[k - 1] + (S2[k - 2] if k >= 2 else 0) + (S4[k - 4] if k >= 4 else 0)
                         for k in range(1, n + 1)]
            if new == S:
                break
            S = new
        return S[n]
    raise ValueError(f"no counting formula for family {family!r}")


def _poly_mul(a: List[int], b: List[int], n: int) -> List[int]:
    r = [0] * (n + 1)
    for i, ai in enumerate(a):
        if ai:
            for j in range(n + 1 - i):
                r[i + j] += ai * b[j]
    return r


# ---------------------------------------------------------------------------
# Exact selection (minimize the sum of deltas); candidates: {(i, j): delta < 0}
# ---------------------------------------------------------------------------
def _compress(cands: Dict[Arc, float]):
    pts = sorted({p for ab in cands for p in ab})
    idx = {p: t for t, p in enumerate(pts)}
    by_left: Dict[int, List[Tuple[int, float]]] = {}
    for (a, b), w in cands.items():
        by_left.setdefault(idx[a], []).append((idx[b], w))
    return pts, idx, by_left


def _noncrossing_table(E: int, by_left):
    """M[i][j]: best noncrossing set inside compressed positions [i, j]; ch = traceback choice."""
    M = [[0.0] * E for _ in range(E)]
    ch: List[List[Optional[int]]] = [[None] * E for _ in range(E)]

    def get(i, j):
        return M[i][j] if i <= j else 0.0

    for i in range(E - 1, -1, -1):
        arcs = by_left.get(i, ())
        Mi1 = M[i + 1] if i + 1 < E else None
        for j in range(i, E):
            best = Mi1[j] if (Mi1 is not None and i + 1 <= j) else 0.0
            bc = None
            for b, w in arcs:
                if b <= j:
                    v = w + get(i + 1, b - 1) + get(b + 1, j)
                    if v < best:
                        best, bc = v, b
            M[i][j] = best
            ch[i][j] = bc
    return M, ch, get


def _trace_noncrossing(ch, i, j, out):
    while i < j:
        b = ch[i][j]
        if b is None:
            i += 1
        else:
            out.append((i, b))
            _trace_noncrossing(ch, i + 1, b - 1, out)
            i = b + 1


def select_dyn(cands: Dict[Arc, float]) -> Tuple[float, List[Arc]]:
    """Pairwise disjoint intervals: k-bonacci's positional DP with k = n."""
    if not cands:
        return 0.0, []
    n = max(b for _a, b in cands) + 1
    value, chosen = solve_dp(n, [(a, b, w) for (a, b), w in cands.items()])
    return float(value), [tuple(x) for x in chosen]


def select_c0(cands: Dict[Arc, float]) -> Tuple[float, List[Arc]]:
    """Noncrossing (Motzkin): interval DP on the compressed endpoints, O(E^2 + E |C|)."""
    if not cands:
        return 0.0, []
    pts, _idx, by_left = _compress(cands)
    E = len(pts)
    M, ch, _get = _noncrossing_table(E, by_left)
    out: List[Arc] = []
    _trace_noncrossing(ch, 0, E - 1, out)
    return M[0][E - 1], sorted((pts[a], pts[b]) for a, b in out)


def _select_h_family(cands: Dict[Arc, float], per_arc: bool, core_k: int):
    """c1 (per_arc=False: at most one crossing pair) or pa1 (per_arc=True: every arc crossed at most once).

    An H-core is a crossing pair (i, b), (c, d), i < c < b < d; its three gaps are independent
    sub-problems (noncrossing for c1, the same family for pa1). Cores come from the core_k most
    improving candidates. Returns (value, arcs, core_cap_binding)."""
    if not cands:
        return 0.0, [], False
    pts, idx, by_left = _compress(cands)
    E = len(pts)
    M, chM, getM = _noncrossing_table(E, by_left)
    core_src = sorted(cands.items(), key=lambda kv: kv[1])
    capped = len(core_src) > core_k
    core_src = sorted((idx[a], idx[b], w) for (a, b), w in core_src[:core_k])
    cores_at: Dict[int, Dict[int, List[Tuple[int, int, float]]]] = {}
    for x in range(len(core_src)):
        i, b, w1 = core_src[x]
        for y in range(x + 1, len(core_src)):
            c, d, w2 = core_src[y]
            if c >= b:
                break
            if i < c < b < d:
                cores_at.setdefault(i, {}).setdefault(d, []).append((b, c, w1 + w2))
    P = [[0.0] * E for _ in range(E)]
    chP: List[List[Optional[tuple]]] = [[None] * E for _ in range(E)]

    def getP(i, j):
        return P[i][j] if i <= j else 0.0

    region = getP if per_arc else getM
    for i in range(E - 1, -1, -1):
        best_core: Dict[int, Tuple[float, Tuple[int, int]]] = {}
        for d, lst in cores_at.get(i, {}).items():
            bv, arg = INF, None
            for b, c, w in lst:
                v = w + region(i + 1, c - 1) + region(c + 1, b - 1) + region(b + 1, d - 1)
                if v < bv:
                    bv, arg = v, (b, c)
            best_core[d] = (bv, arg)
        arcs = by_left.get(i, ())
        for j in range(i, E):
            best, arg = getP(i + 1, j), ("skip",)
            for b, w in arcs:
                if b > j:
                    continue
                if per_arc:
                    v = w + getP(i + 1, b - 1) + getP(b + 1, j)
                    if v < best:
                        best, arg = v, ("arc", b, None)
                else:
                    v1 = w + getP(i + 1, b - 1) + getM(b + 1, j)
                    v2 = w + getM(i + 1, b - 1) + getP(b + 1, j)
                    if v1 < best:
                        best, arg = v1, ("arc", b, "in")
                    if v2 < best:
                        best, arg = v2, ("arc", b, "out")
            for d, (kv, karg) in best_core.items():
                if d <= j:
                    v = kv + (getP(d + 1, j) if per_arc else getM(d + 1, j))
                    if v < best:
                        best, arg = v, ("core", d, karg)
            P[i][j] = best
            chP[i][j] = arg
    out: List[Arc] = []

    def trace_m(i, j):
        _trace_noncrossing(chM, i, j, out)

    def trace_p(i, j):
        while i < j:
            a = chP[i][j]
            if a[0] == "skip":
                i += 1
                continue
            if a[0] == "arc":
                b = a[1]
                out.append((i, b))
                if per_arc:
                    trace_p(i + 1, b - 1)
                    i = b + 1
                    continue
                if a[2] == "in":
                    trace_p(i + 1, b - 1)
                    trace_m(b + 1, j)
                else:
                    trace_m(i + 1, b - 1)
                    trace_p(b + 1, j)
                return
            d, (b, c) = a[1], a[2]
            out.append((i, b))
            out.append((c, d))
            inner = trace_p if per_arc else trace_m
            inner(i + 1, c - 1)
            inner(c + 1, b - 1)
            inner(b + 1, d - 1)
            if per_arc:
                i = d + 1
                continue
            trace_m(d + 1, j)
            return

    trace_p(0, E - 1)
    return P[0][E - 1], sorted((pts[a], pts[b]) for a, b in out), capped


def select_c1(cands: Dict[Arc, float], core_k: int = CORE_K):
    v, arcs, _ = _select_h_family(cands, per_arc=False, core_k=core_k)
    return v, arcs


def select_pa1(cands: Dict[Arc, float], core_k: int = CORE_K):
    v, arcs, _ = _select_h_family(cands, per_arc=True, core_k=core_k)
    return v, arcs


def select_inv(cands: Dict[Arc, float]) -> Tuple[float, List[Arc]]:
    """Any involution: maximum-weight matching (Edmonds' blossom, networkx), weights = -delta."""
    if not cands:
        return 0.0, []
    import networkx as nx  # local import: only this family needs networkx

    G = nx.Graph()
    for (a, b), w in cands.items():
        G.add_edge(a, b, weight=-w)
    arcs = sorted(tuple(sorted(e)) for e in nx.max_weight_matching(G, maxcardinality=False))
    return float(sum(cands[a] for a in arcs)), arcs


# ---------------------------------------------------------------------------
# Neighborhood
# ---------------------------------------------------------------------------
def parse_crossing_mode(mode: str):
    """'cross_c0_safe_cp' -> ('c0', 'safe', 'critical_path'); None if not a crossing mode."""
    if not mode.startswith("cross_"):
        return None
    parts = mode[len("cross_"):].split("_")
    boundary = "last_machine"
    if parts and parts[-1] == "cp":
        boundary = "critical_path"
        parts = parts[:-1]
    rule = "plain"
    if parts and parts[-1] in ("safe", "prefix"):
        rule = parts[-1]
        parts = parts[:-1]
    family = "_".join(parts)
    if family not in FAMILIES:
        raise ValueError(f"unknown crossing family in mode {mode!r} (known: {FAMILIES})")
    if family == "swap" and rule != "plain":
        raise ValueError("cross_swap has no acceptance-rule suffix")
    return family, rule, boundary


def _apply(pi: List[int], arcs: List[Arc]) -> List[int]:
    q = pi.copy()
    for i, j in arcs:
        q[i], q[j] = q[j], q[i]
    return q


def crossing_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    family: str,
    rule: str = "plain",
    boundary_mode: str = "last_machine",
    top_k: int = TOP_K,
    core_k: int = CORE_K,
    stats: Optional[dict] = None,
) -> Tuple[List[int], int, List[Arc]]:
    """Best compound move of one family under one acceptance rule. Returns (new_pi, new_cmax, swaps)."""
    t0 = time.perf_counter()
    n = len(pi)
    base, cand_list = compute_candidates(pi, processing_times, max(2, n), True, boundary_mode)
    if not cand_list:
        return pi.copy(), base, []
    best_single: Candidate = min(cand_list, key=lambda c: c[2])
    improving = {(i, j): d for i, j, d in cand_list if d < 0}
    _stats["candidates"] += len(cand_list)
    _stats["improving"] += len(improving)
    capped_top = capped_core = False
    pred = 0.0
    ts = time.perf_counter()
    if family == "swap" or not improving:
        chosen: List[Arc] = []
    elif family == "dyn":
        pred, chosen = select_dyn(improving)
    else:
        sel = improving
        if family != "dyn" and len(sel) > top_k:
            sel = dict(sorted(sel.items(), key=lambda kv: kv[1])[:top_k])
            capped_top = True
        if family == "c0":
            pred, chosen = select_c0(sel)
        elif family in ("c1", "pa1"):
            pred, chosen, capped_core = _select_h_family(sel, per_arc=(family == "pa1"), core_k=core_k)
        elif family == "inv":
            pred, chosen = select_inv(sel)
        else:
            raise ValueError(family)
    _stats["select_ms"] += (time.perf_counter() - ts) * 1000
    if not improving:
        _stats["no_improving"] += 1
    if not chosen:
        chosen = [(best_single[0], best_single[1])]
        pred = best_single[2]
    new_pi = _apply(pi, chosen)
    new_c = c_max(new_pi, processing_times)
    fallback = False
    if rule in ("safe", "prefix") and len(chosen) > 1:
        single_c = base + best_single[2]          # a single-swap delta is exact
        best_arcs, best_c, best_pi = [(best_single[0], best_single[1])], single_c, None
        if new_c <= single_c:
            best_arcs, best_c, best_pi = chosen, new_c, new_pi
        if rule == "prefix":
            delta = {(i, j): d for i, j, d in cand_list}
            order = sorted(chosen, key=lambda ab: delta[ab])
            for k in range(2, len(order)):        # k = 1 is dominated by the best single swap, k = len is done
                q = _apply(pi, order[:k])
                qc = c_max(q, processing_times)
                if qc < best_c:
                    best_arcs, best_c, best_pi = order[:k], qc, q
        if best_arcs is not chosen:
            fallback = True
            chosen = best_arcs
            new_c = best_c
            new_pi = best_pi if best_pi is not None else _apply(pi, chosen)
    _stats["moves"] += 1
    _stats["swaps"] += len(chosen)
    _stats["crossings"] += crossings(chosen)
    _stats["nestings"] += nestings(chosen)
    _stats["fallback"] += int(fallback)
    _stats["cap_top"] += int(capped_top)
    _stats["cap_core"] += int(capped_core)
    _stats["pred_sum"] += pred
    _stats["real_sum"] += new_c - base
    _stats["total_ms"] += (time.perf_counter() - t0) * 1000
    if stats is not None:
        stats.update({"family": family, "rule": rule, "base_cmax": base, "num_candidates": len(cand_list),
                      "num_improving": len(improving), "predicted_delta": pred, "real_delta": new_c - base,
                      "num_swaps": len(chosen), "crossings": crossings(chosen), "fallback": fallback,
                      "cap_top": capped_top, "cap_core": capped_core})
    return new_pi, new_c, sorted(chosen)
