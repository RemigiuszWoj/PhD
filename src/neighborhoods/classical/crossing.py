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
Plan v4 options (any order after the family; articles/crossings/PLAN.md, note v4):
    _k<K>      equal cardinality: the selected move is cut to its K most improving swaps (still in the family,
               the families are hereditary)
    _top<K>    exact acceptance among K moves: the selected move plus K-1 alternatives, alternative t is the
               selection with the t-th most improving arc of the first move removed; the best (real C_max) of
               these and of the best single swap is applied (_top1 = _safe)
    _lim<T>x<C>  caps TOP_K = T and CORE_K = C (cap-sensitivity ablation)
    cross_q<K>  second-order (pair-corrected) QUBO over the K most improving candidates: linear terms are the
               exact single-swap deltas, quadratic terms the measured pair interactions
               C(pi o a o b) - C(pi) - delta_a - delta_b, swaps sharing an endpoint are forbidden; solved
               exactly by enumeration (K <= 20). This is the QUBO sent to the annealer / QAOA in E4.

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

import re
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
              "select_ms", "total_ms", "pair_evals", "exact_evals")
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
Q_MAX = 20  # enumeration limit of the pair-corrected QUBO (cross_q<K>)


def parse_crossing_options(mode: str) -> Optional[dict]:
    """'cross_inv_k4_cp' -> {'family': 'inv', 'rule': 'plain', 'boundary': 'critical_path', 'card': 4, ...};
    None if not a crossing mode. Keys: family, rule, boundary, card, top, top_k, core_k, q."""
    if not mode.startswith("cross_"):
        return None
    head, *opts = mode[len("cross_"):].split("_")
    o = {"family": head, "rule": "plain", "boundary": "last_machine", "card": None, "top": None,
         "top_k": TOP_K, "core_k": CORE_K, "q": None}
    if re.fullmatch(r"q\d+", head):
        o["family"], o["q"] = "q", int(head[1:])
        if not 2 <= o["q"] <= Q_MAX:
            raise ValueError(f"cross_q<K> needs 2 <= K <= {Q_MAX}, got {mode!r}")
    elif head not in FAMILIES:
        raise ValueError(f"unknown crossing family in mode {mode!r} (known: {FAMILIES} and q<K>)")
    for t in opts:
        if t == "cp":
            o["boundary"] = "critical_path"
        elif t in ("safe", "prefix"):
            o["rule"] = t
        elif m := re.fullmatch(r"k(\d+)", t):
            o["card"] = int(m[1])
        elif m := re.fullmatch(r"top(\d+)", t):
            o["top"] = int(m[1])
        elif m := re.fullmatch(r"lim(\d+)x(\d+)", t):
            o["top_k"], o["core_k"] = int(m[1]), int(m[2])
        else:
            raise ValueError(f"unknown option {t!r} in crossing mode {mode!r}")
    if o["family"] in ("swap", "q") and (o["rule"] != "plain" or o["card"] or o["top"]):
        raise ValueError(f"{mode!r}: cross_swap and cross_q<K> take no rule, _k or _top option")
    if o["top"] is not None and (o["top"] < 1 or o["rule"] != "plain"):
        raise ValueError(f"{mode!r}: _top<K> needs K >= 1 and no _safe/_prefix")
    if o["card"] is not None and o["card"] < 1:
        raise ValueError(f"{mode!r}: _k<K> needs K >= 1")
    return o


def parse_crossing_mode(mode: str):
    """'cross_c0_safe_cp' -> ('c0', 'safe', 'critical_path'); None if not a crossing mode.
    Plan v4 options (_k, _top, _lim, cross_q<K>) are parsed by parse_crossing_options."""
    o = parse_crossing_options(mode)
    return None if o is None else (o["family"], o["rule"], o["boundary"])


# ---------------------------------------------------------------------------
# Second-order (pair-corrected) QUBO, plan v4
# ---------------------------------------------------------------------------
def pair_corrected_qubo(pi: List[int], processing_times: List[List[int]], base: int,
                        cands: Dict[Arc, float], K: int):
    """QUBO over the K most improving candidates: (arcs, linear, quad, conflicts).
    linear[x] = exact delta of arc x; quad[(x, y)] = C(pi o x o y) - base - delta_x - delta_y for arcs with
    distinct endpoints; conflicts = pairs sharing an endpoint (forbidden, not evaluated)."""
    arcs = [a for a, _ in sorted(cands.items(), key=lambda kv: kv[1])[:K]]
    linear = [float(cands[a]) for a in arcs]
    quad: Dict[Tuple[int, int], float] = {}
    conflicts: List[Tuple[int, int]] = []
    for x in range(len(arcs)):
        for y in range(x + 1, len(arcs)):
            a, b = arcs[x], arcs[y]
            if len({*a, *b}) < 4:
                conflicts.append((x, y))
                continue
            q = _apply(pi, [a, b])
            quad[(x, y)] = float(c_max(q, processing_times) - base - linear[x] - linear[y])
    return arcs, linear, quad, conflicts


def solve_qubo_exact(linear: List[float], quad: Dict[Tuple[int, int], float],
                     conflicts: List[Tuple[int, int]]) -> Tuple[float, List[int]]:
    """Minimum energy over conflict-free subsets (matchings), depth-first enumeration; K <= Q_MAX."""
    K = len(linear)
    if K > Q_MAX:
        raise ValueError(f"exact QUBO enumeration is limited to {Q_MAX} variables, got {K}")
    W = [[0.0] * K for _ in range(K)]
    for (x, y), w in quad.items():
        W[x][y] = W[y][x] = w
    bad = [set() for _ in range(K)]
    for x, y in conflicts:
        bad[x].add(y)
        bad[y].add(x)
    best = [0.0, []]
    chosen: List[int] = []

    def rec(x: int, energy: float) -> None:
        if x == K:
            if energy < best[0] - 1e-12:
                best[0], best[1] = energy, chosen.copy()
            return
        rec(x + 1, energy)                                   # x out
        if not any(y in bad[x] for y in chosen):             # x in
            e = energy + linear[x] + sum(W[x][y] for y in chosen)
            chosen.append(x)
            rec(x + 1, e)
            chosen.pop()

    rec(0, 0.0)
    return best[0], best[1]


def _alternatives(sel: Dict[Arc, float], first: List[Arc], count: int, select) -> List[List[Arc]]:
    """count - 1 alternative moves: the selection with the t-th most improving arc of `first` removed."""
    out = []
    for a in sorted(first, key=lambda ab: sel[ab])[:max(0, count - 1)]:
        rest = {k: v for k, v in sel.items() if k != a}
        _, arcs = select(rest)
        if arcs and sorted(arcs) not in (sorted(m) for m in out) and sorted(arcs) != sorted(first):
            out.append(arcs)
    return out


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
    card: Optional[int] = None,
    top: Optional[int] = None,
    q: Optional[int] = None,
) -> Tuple[List[int], int, List[Arc]]:
    """Best compound move of one family under one acceptance rule. Returns (new_pi, new_cmax, swaps).
    card / top / q: plan v4 options (_k<K>, _top<K>, cross_q<K>), see the module docstring."""
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

    def select(c: Dict[Arc, float]) -> Tuple[float, List[Arc]]:
        nonlocal capped_top, capped_core
        if family == "dyn":
            return select_dyn(c)
        if len(c) > top_k:
            c = dict(sorted(c.items(), key=lambda kv: kv[1])[:top_k])
            capped_top = True
        if family == "c0":
            return select_c0(c)
        if family in ("c1", "pa1"):
            pr, arcs, cc = _select_h_family(c, per_arc=(family == "pa1"), core_k=core_k)
            capped_core = capped_core or cc
            return pr, arcs
        if family == "inv":
            return select_inv(c)
        raise ValueError(family)

    def cut(arcs: List[Arc]) -> List[Arc]:
        if card is None or len(arcs) <= card:
            return arcs
        return sorted(arcs, key=lambda ab: improving[ab])[:card]

    alternatives: List[List[Arc]] = []
    if family == "swap" or not improving:
        chosen: List[Arc] = []
    elif family == "q":
        arcs_q, lin, quad, conf = pair_corrected_qubo(pi, processing_times, base, improving, q)
        _stats["pair_evals"] += len(quad)
        pred, idx = solve_qubo_exact(lin, quad, conf)
        chosen = [arcs_q[x] for x in idx]
    else:
        pred, chosen = select(improving)
        chosen = cut(chosen)
        if top is not None and top > 1 and chosen:
            alternatives = [cut(m) for m in _alternatives(improving, chosen, top, select)]
        if card is not None:
            pred = float(sum(improving[ab] for ab in chosen))
    _stats["select_ms"] += (time.perf_counter() - ts) * 1000
    if not improving:
        _stats["no_improving"] += 1
    if not chosen:
        chosen = [(best_single[0], best_single[1])]
        pred = best_single[2]
    new_pi = _apply(pi, chosen)
    new_c = c_max(new_pi, processing_times)
    fallback = False
    if top is not None:
        # exact acceptance among the first move, its alternatives and the best single swap (_top1 = _safe)
        single = [(best_single[0], best_single[1])]
        best_arcs, best_c, best_pi = chosen, new_c, new_pi
        for arcs in alternatives + [single]:
            if sorted(arcs) == sorted(best_arcs):
                continue
            qq = _apply(pi, arcs)
            qc = c_max(qq, processing_times) if arcs is not single else base + best_single[2]
            if qc < best_c:
                best_arcs, best_c, best_pi = arcs, qc, qq
        _stats["exact_evals"] += len(alternatives) + 1
        if best_arcs is not chosen:
            fallback = True
            chosen, new_c, new_pi = best_arcs, best_c, best_pi
    if rule in ("safe", "prefix") and len(chosen) > 1:
        single_c = base + best_single[2]          # a single-swap delta is exact
        best_arcs, best_c, best_pi = [(best_single[0], best_single[1])], single_c, None
        if new_c <= single_c:
            best_arcs, best_c, best_pi = chosen, new_c, new_pi
        if rule == "prefix":
            delta = {(i, j): d for i, j, d in cand_list}
            order = sorted(chosen, key=lambda ab: delta[ab])
            for k in range(2, len(order)):        # k = 1 is dominated by the best single swap, k = len is done
                qq = _apply(pi, order[:k])
                qc = c_max(qq, processing_times)
                if qc < best_c:
                    best_arcs, best_c, best_pi = order[:k], qc, qq
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
