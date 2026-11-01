"""Counting involutions (partial matchings) of [n] by crossings.

Exact counts via the Flajolet J-fraction: Motzkin paths, up/level weight 1,
down step from height h weight [h]_q = 1 + q + ... + q^(h-1) (q marks crossings).
Brute force for n <= 12 to validate, plus variants:
  per-arc   : every arc crossed by at most one other arc
  3nc       : no three mutually crossing arcs (3-noncrossing)
  nonnesting: no nested pair
"""
import itertools, math, sys
from functools import lru_cache

def involutions_arcs(n):
    """Yield involutions of range(n) as lists of arcs (a,b), a<b."""
    def rec(free, arcs):
        if not free:
            yield list(arcs); return
        a = free[0]; rest = free[1:]
        yield from rec(rest, arcs)                      # a fixed
        for idx, b in enumerate(rest):
            arcs.append((a, b))
            yield from rec(rest[:idx] + rest[idx+1:], arcs)
            arcs.pop()
    yield from rec(list(range(n)), [])

def cross(x, y):
    (a, b), (c, d) = sorted([x, y])
    return a < c < b < d

def nest(x, y):
    (a, b), (c, d) = sorted([x, y])
    return a < c < d < b

def brute(n):
    tot = 0; bycr = {}; perarc1 = 0; nc3 = 0; nonnest = 0; maxcr = 0
    for arcs in involutions_arcs(n):
        tot += 1
        k = len(arcs)
        cnt = [0]*k; cr = 0; ne = 0
        for i in range(k):
            for j in range(i+1, k):
                if cross(arcs[i], arcs[j]):
                    cr += 1; cnt[i] += 1; cnt[j] += 1
                elif nest(arcs[i], arcs[j]):
                    ne += 1
        bycr[cr] = bycr.get(cr, 0) + 1
        maxcr = max(maxcr, cr)
        if all(c <= 1 for c in cnt): perarc1 += 1
        if ne == 0: nonnest += 1
        # 3-crossing: three pairwise crossing arcs
        has3 = False
        if cr >= 3:
            for t in itertools.combinations(range(k), 3):
                i, j, l = t
                if cross(arcs[i], arcs[j]) and cross(arcs[i], arcs[l]) and cross(arcs[j], arcs[l]):
                    has3 = True; break
        if not has3: nc3 += 1
    return tot, bycr, perarc1, nc3, nonnest, maxcr

def jfrac_counts(N, C):
    """cnt[n][c] = #involutions of [n] with exactly c crossings, c <= C (exact ints)."""
    # state: dict height -> poly (list len C+1)
    res = []
    cur = {0: [1] + [0]*C}
    res.append(cur[0][:])
    for n in range(1, N+1):
        nxt = {}
        for h, poly in cur.items():
            # level
            acc = nxt.setdefault(h, [0]*(C+1))
            for i in range(C+1): acc[i] += poly[i]
            # up
            acc = nxt.setdefault(h+1, [0]*(C+1))
            for i in range(C+1): acc[i] += poly[i]
            # down with [h]_q
            if h > 0:
                acc = nxt.setdefault(h-1, [0]*(C+1))
                for i in range(C+1):
                    if poly[i]:
                        for j in range(min(h, C+1-i)):
                            acc[i+j] += poly[i]
        # prune heights that cannot return to 0
        cur = {h: p for h, p in nxt.items() if h <= N - n}
        res.append(cur.get(0, [0]*(C+1))[:])
    return res

def involution_numbers(N):
    I = [1, 1]
    for n in range(2, N+1): I.append(I[-1] + (n-1)*I[-2])
    return I

def motzkin(N):
    M = [1, 1]
    for n in range(2, N+1):
        M.append(((2*n+1)*M[-1] + (3*n-3)*M[-2])//(n+2))
    return M

if __name__ == "__main__":
    print("== brute force vs J-fraction (n<=11) ==")
    J = jfrac_counts(12, 12)
    I = involution_numbers(600); M = motzkin(600)
    for n in range(1, 12):
        tot, bycr, pa1, nc3, nn, mx = brute(n)
        le = lambda c: sum(v for k, v in bycr.items() if k <= c)
        jl = lambda c: sum(J[n][:c+1])
        ok = all(le(c) == jl(c) for c in range(0, 6))
        print(f"n={n:2d} inv={tot:6d} I={I[n]:6d} | <=0:{le(0):5d} M={M[n]:5d} <=1:{le(1):6d} <=2:{le(2):6d} <=3:{le(3):6d} "
              f"| perarc<=1:{pa1:6d} 3-noncross:{nc3:6d} nonnest:{nn:5d} maxcr={mx} C(n//2,2)={math.comb(n//2,2)} Jok={ok}")
