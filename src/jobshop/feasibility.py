"""Feasibility of reorderings inside critical blocks (theorem T2, articles/jobshop/PLAN.md §4).

T2: let sigma' reorder operations only inside the blocks of one critical path of a
feasible sigma. sigma' is infeasible  <=>  some block B has operations x, y with x before
y in sigma, y before x in sigma', and an E-path x -> y in G(sigma): x -> JS(x) ~> JP(y) -> y
(or JS(x) = JP(y)) that does not pass through other operations of B.

For a single endpoint swap, a single insertion (rotation) and laminar (non-crossing)
families, the plain reachability JS(x) ~> JP(y) gives the same verdict; for arbitrary
block permutations only the strict version is exact (pilot: 1 / 70 680 false alarms of
the plain test on full block shuffles, 0 for the strict one).
"""

from __future__ import annotations

from src.jobshop.instance import Instance
from src.jobshop.schedule import Eval


def reach(inst: Instance, ev: Eval, src: int, dst: int, forbid: set | None = None) -> bool:
    """Is dst reachable from src in G(sigma)? DFS pruned by heads (a path a ~> b needs r(b) > r(a))."""
    if src == dst:
        return True
    if forbid and src in forbid:
        return False
    m, r = inst.m, ev.r
    lim = r[dst]
    stack, seen = [src], {src}
    while stack:
        u = stack.pop()
        for v in ((u + 1) if (u + 1) % m else -1, ev.msucc[u]):
            if v < 0 or v in seen or r[v] > lim or (forbid and v in forbid and v != dst):
                continue
            if v == dst:
                return True
            seen.add(v)
            stack.append(v)
    return False


def forbidden_pair(inst: Instance, ev: Eval, x: int, y: int, block_ops: set | None) -> bool:
    """E-path x -> y exists (x before y in the same critical block)."""
    m, r = inst.m, ev.r
    if (x + 1) % m == 0 or y % m == 0:
        return False
    js, jp = x + 1, y - 1
    if js != jp and r[jp] <= r[js]:
        return False  # heads certificate: no path js ~> jp
    return reach(inst, ev, js, jp, forbid=block_ops)


def order_feasible(inst: Instance, ev: Eval, old: list[int], new: list[int]) -> bool:
    """Feasibility of replacing a contiguous stretch `old` of one critical block by `new` (T2).

    Pairs adjacent in `old` are never forbidden (an E-path would be longer than the
    critical arc; T3a), so only pairs at distance >= 2 are tested.
    """
    npos = {o: t for t, o in enumerate(new)}
    block_ops = set(old)
    for a in range(len(old)):
        x = old[a]
        for b in range(a + 2, len(old)):
            y = old[b]
            if npos[y] < npos[x] and forbidden_pair(inst, ev, x, y, block_ops):
                return False
    return True


def blocks_reorder_feasible(inst: Instance, ev: Eval, path: list[int], blks, new_orders) -> bool:
    """T2 for arbitrary permutations of several blocks of one critical path at once."""
    for (s, L), new in zip(blks, new_orders):
        old = path[s : s + L]
        if new != old and not order_feasible(inst, ev, old, new):
            return False
    return True
