"""Exact selection of the best compound move (min sum of deltas) for the crossing family.

Candidates: dict {(a, b): w} with a < b, w < 0 (only improving arcs matter: the
families are hereditary, so an arc with w >= 0 never helps the predicted optimum).

  select_c0      : noncrossing (Motzkin), interval DP
  select_c1      : <= 1 crossing, interval DP + 'H-core' table K[i][d]
  select_perarc1 : every arc crossed at most once (any number of H-cores)
  select_c2      : <= 2 crossings (two H-cores, or one 3-arc core with 2 crossings)
  select_inv     : any involution (max-weight matching, networkx blossom)
  select_milp    : <= c crossings for any c (scipy milp / HiGHS)
All DPs work on the compressed set of candidate endpoints (positions that are no
candidate's endpoint are irrelevant), so cost depends on |C|, not on n.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root
RES = HERE / "measurements"
import itertools
import networkx as nx
import numpy as np

INF = float("inf")

def _compress(cands):
    pts = sorted({p for ab in cands for p in ab})
    idx = {p: t for t, p in enumerate(pts)}
    W = {}
    for (a, b), w in cands.items():
        W[(idx[a], idx[b])] = w
    return pts, W

def _by_left(W):
    L = {}
    for (a, b), w in W.items():
        L.setdefault(a, []).append((b, w))
    return L

def _M_table(E, byL):
    """M[i][j] best noncrossing within [i, j] (compressed coords), with choice for traceback."""
    M = [[0.0] * (E + 1) for _ in range(E + 1)]   # M[i][j] for i<=j ; index j+1 sentinel handled
    ch = [[None] * (E + 1) for _ in range(E + 1)]
    def get(i, j):
        return M[i][j] if i <= j else 0.0
    for i in range(E - 1, -1, -1):
        for j in range(i, E):
            best = get(i + 1, j); bc = None
            for b, w in byL.get(i, ()):
                if b <= j:
                    v = w + get(i + 1, b - 1) + get(b + 1, j)
                    if v < best:
                        best = v; bc = b
            M[i][j] = best; ch[i][j] = bc
    return M, ch, get

def _trace_M(ch, i, j, out):
    while i < j:
        b = ch[i][j]
        if b is None:
            i += 1
        else:
            out.append((i, b)); _trace_M(ch, i + 1, b - 1, out); i = b + 1

def select_c0(cands):
    if not cands: return 0.0, []
    pts, W = _compress(cands); E = len(pts); byL = _by_left(W)
    M, ch, get = _M_table(E, byL)
    out = []; _trace_M(ch, 0, E - 1, out)
    return M[0][E - 1], [(pts[a], pts[b]) for a, b in out]

def _cores_1(W):
    """All crossing pairs (i,b),(c,d) with i<c<b<d, grouped by (i, d)."""
    arcs = sorted(W.items())
    cores = {}
    for (i, b), w1 in arcs:
        for (c, d), w2 in arcs:
            if i < c < b < d:
                cores.setdefault((i, d), []).append((b, c, w1 + w2))
    return cores

def _select_one_crossing_family(cands, unlimited):
    """unlimited=False -> <=1 crossing ; True -> per-arc <= 1 crossing."""
    if not cands: return 0.0, []
    pts, W = _compress(cands); E = len(pts); byL = _by_left(W)
    M, chM, getM = _M_table(E, byL)
    cores = _cores_1(W)
    # K[i][d] = best H-core spanning exactly [i, d]
    # inner regions use M (for <=1) or Q (for per-arc) -> for per-arc we need Q first; do DP by interval length
    P = [[0.0] * (E + 1) for _ in range(E + 1)]
    chP = [[None] * (E + 1) for _ in range(E + 1)]
    def getP(i, j):
        return P[i][j] if i <= j else 0.0
    R = getP if unlimited else getM          # regions inside a core
    coresL = {}
    for (i, d), lst in cores.items():
        coresL.setdefault(i, []).append((d, lst))
    for i in range(E - 1, -1, -1):
        # K for this i depends on regions strictly inside -> already computed (larger i)
        Kd = {}
        for d, lst in coresL.get(i, ()):
            best = INF; arg = None
            for b, c, w in lst:
                v = w + R(i + 1, c - 1) + R(c + 1, b - 1) + R(b + 1, d - 1)
                if v < best: best = v; arg = (b, c)
            Kd[d] = (best, arg)
        for j in range(i, E):
            best = getP(i + 1, j); arg = ("skip",)
            for b, w in byL.get(i, ()):
                if b <= j:
                    if unlimited:
                        v = w + getP(i + 1, b - 1) + getP(b + 1, j)
                        if v < best: best = v; arg = ("arc", b, None)
                    else:
                        v1 = w + getP(i + 1, b - 1) + getM(b + 1, j)
                        v2 = w + getM(i + 1, b - 1) + getP(b + 1, j)
                        if v1 < best: best = v1; arg = ("arc", b, "in")
                        if v2 < best: best = v2; arg = ("arc", b, "out")
            for d, (kv, karg) in Kd.items():
                if d <= j:
                    v = kv + (getP(d + 1, j) if unlimited else getM(d + 1, j))
                    if v < best: best = v; arg = ("core", d, karg)
            P[i][j] = best; chP[i][j] = arg
    # traceback
    out = []
    def trM(i, j): _trace_M(chM, i, j, out)
    def trP(i, j):
        while i < j:
            a = chP[i][j]
            if a[0] == "skip": i += 1; continue
            if a[0] == "arc":
                b = a[1]; out.append((i, b))
                if unlimited: trP(i + 1, b - 1); i = b + 1; continue
                if a[2] == "in": trP(i + 1, b - 1); trM(b + 1, j)
                else: trM(i + 1, b - 1); trP(b + 1, j)
                return
            d, (b, c) = a[1], a[2]
            out.append((i, b)); out.append((c, d))
            tr = trP if unlimited else trM
            tr(i + 1, c - 1); tr(c + 1, b - 1); tr(b + 1, d - 1)
            i = d + 1
            if not unlimited: trM(i, j); return
    trP(0, E - 1)
    return P[0][E - 1], sorted((pts[a], pts[b]) for a, b in out)

def select_c1(cands): return _select_one_crossing_family(cands, False)
def select_perarc1(cands): return _select_one_crossing_family(cands, True)

def select_dyn(cands):
    """Dynasearch: best set of pairwise disjoint intervals [a, b] (= k-bonacci with k = n), weighted interval scheduling."""
    if not cands: return 0.0, []
    n = max(b for a, b in cands) + 1
    ending = {}
    for (a, b), w in cands.items(): ending.setdefault(b, []).append((a, w))
    best = [0.0] * (n + 1); ch = [None] * (n + 1)
    for q in range(1, n + 1):
        best[q] = best[q - 1]
        for a, w in ending.get(q - 1, ()):
            if best[a] + w < best[q]: best[q] = best[a] + w; ch[q] = (a, q - 1)
    out, q = [], n
    while q > 0:
        if ch[q] is None: q -= 1
        else: out.append(ch[q]); q = ch[q][0]
    return best[n], sorted(out)

def select_inv(cands):
    G = nx.Graph()
    for (a, b), w in cands.items():
        G.add_edge(a, b, weight=-w)
    mt = nx.max_weight_matching(G, maxcardinality=False)
    arcs = sorted(tuple(sorted(e)) for e in mt)
    return sum(cands[a] for a in arcs), arcs

def crossings(arcs):
    arcs = sorted(arcs); cr = 0
    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if c < b < d: cr += 1
    return cr

def max_cross_per_arc(arcs):
    arcs = sorted(arcs); cnt = [0] * len(arcs)
    for x in range(len(arcs)):
        a, b = arcs[x]
        for y in range(x + 1, len(arcs)):
            c, d = arcs[y]
            if c < b < d: cnt[x] += 1; cnt[y] += 1
    return max(cnt) if cnt else 0

def select_milp(cands, c, time_limit=60):
    """<= c crossings via MILP (HiGHS). c=None -> no crossing limit (matching)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    arcs = sorted(cands); A = len(arcs)
    if A == 0: return 0.0, []
    crossP = [(x, y) for x in range(A) for y in range(x + 1, A)
              if (arcs[x][0] < arcs[y][0] < arcs[x][1] < arcs[y][1]) or (arcs[y][0] < arcs[x][0] < arcs[y][1] < arcs[x][1])]
    nY = len(crossP) if c is not None else 0
    nv = A + nY
    obj = np.zeros(nv); obj[:A] = [cands[a] for a in arcs]
    rows = []; lb = []; ub = []
    pts = sorted({p for ab in arcs for p in ab})
    for p in pts:
        r = np.zeros(nv)
        for x, (a, b) in enumerate(arcs):
            if a == p or b == p: r[x] = 1
        rows.append(r); lb.append(-np.inf); ub.append(1)
    if c is not None:
        for t, (x, y) in enumerate(crossP):
            r = np.zeros(nv); r[x] = 1; r[y] = 1; r[A + t] = -1
            rows.append(r); lb.append(-np.inf); ub.append(1)
        r = np.zeros(nv); r[A:] = 1; rows.append(r); lb.append(-np.inf); ub.append(c)
    res = milp(obj, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options={"time_limit": time_limit})
    xs = res.x[:A] > 0.5
    chosen = [arcs[x] for x in range(A) if xs[x]]
    return sum(cands[a] for a in chosen), chosen

def brute_best(cands, pred):
    arcs = sorted(cands); best = 0.0; arg = []
    for r in range(1, len(arcs) + 1):
        for S in itertools.combinations(arcs, r):
            ends = [p for ab in S for p in ab]
            if len(set(ends)) < len(ends): continue
            if not pred(S): continue
            v = sum(cands[a] for a in S)
            if v < best: best = v; arg = list(S)
    return best, arg

if __name__ == "__main__":
    import random
    rng = random.Random(7)
    fails = 0; trials = 0
    for t in range(400):
        n = rng.randint(4, 10)
        allp = [(a, b) for a in range(n) for b in range(a + 1, n)]
        k = rng.randint(1, min(len(allp), 11))
        cands = {ab: -rng.randint(1, 20) for ab in rng.sample(allp, k)}
        checks = [
            (select_c0, lambda S: crossings(S) == 0),
            (select_c1, lambda S: crossings(S) <= 1),
            (select_perarc1, lambda S: max_cross_per_arc(S) <= 1),
            (select_inv, lambda S: True),
            (lambda C: select_milp(C, 2), lambda S: crossings(S) <= 2),
            (select_dyn, lambda S: all(x[1] < y[0] for x, y in zip(sorted(S), sorted(S)[1:]))),
        ]
        for f, pred in checks:
            trials += 1
            v, arcs = f(cands)
            bv, _ = brute_best(cands, pred)
            okv = abs(v - bv) < 1e-9 and abs(sum(cands[a] for a in arcs) - v) < 1e-9 and pred(arcs)
            ends = [p for ab in arcs for p in ab]
            okv = okv and len(set(ends)) == len(ends)
            if not okv:
                fails += 1
                if fails < 5: print("FAIL", f, cands, v, bv, arcs)
    print(f"validation: {trials} checks, {fails} failures")
