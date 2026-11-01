"""Disjunctive-graph evaluation, critical path, blocks, start solutions, kick.

A solution is a list of m machine sequences (operation ids). It is feasible iff the
disjunctive graph (job arcs o -> o+1, machine arcs) is acyclic.
"""

from __future__ import annotations

import random

from src.jobshop.instance import Instance


class Eval:
    """Heads r (longest path to o, without p_o), tails q (from o, without p_o), C_max."""

    __slots__ = ("C", "r", "q", "mpred", "msucc", "pos")

    def __init__(self, C, r, q, mpred, msucc, pos):
        self.C, self.r, self.q, self.mpred, self.msucc, self.pos = C, r, q, mpred, msucc, pos


def evaluate(inst: Instance, seq: list[list[int]], need_tails: bool = True) -> Eval | None:
    """O(nm) topological sweep; None if the graph has a cycle (infeasible)."""
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
                best = q[u + 1] + dur[u + 1]
            v = msucc[u]
            if v >= 0 and q[v] + dur[v] > best:
                best = q[v] + dur[v]
            q[u] = best
    return Eval(C, r, q, mpred, msucc, pos)


def cmax(inst: Instance, seq) -> int | None:
    ev = evaluate(inst, seq, need_tails=False)
    return None if ev is None else ev.C


def critical_path(inst: Instance, ev: Eval, prefer: str = "machine") -> list[int]:
    """One critical path, traced backwards from the first operation that ends at C_max.

    Ties (both predecessors critical) go to `prefer` ("machine": longer blocks; "job").
    """
    m, dur, r = inst.m, inst.dur, ev.r
    u = next(o for o in range(inst.N) if r[o] + dur[o] == ev.C)
    path = [u]
    while r[u] > 0:
        jp = u - 1 if u % m else -1
        mp = ev.mpred[u]
        j_ok = jp >= 0 and r[jp] + dur[jp] == r[u]
        m_ok = mp >= 0 and r[mp] + dur[mp] == r[u]
        if j_ok and m_ok:
            u = mp if prefer == "machine" else jp
        else:
            u = jp if j_ok else mp
        path.append(u)
    path.reverse()
    return path


def blocks_of(path: list[int], ev: Eval) -> list[tuple[int, int]]:
    """Blocks as (start index in path, length): maximal runs joined by machine arcs."""
    out = []
    s = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or ev.msucc[path[t - 1]] != path[t]:
            out.append((s, t - s))
            s = t
    return out


# ---------------------------------------------------------------------------
# Start solutions and perturbation
# ---------------------------------------------------------------------------


def decode_opseq(inst: Instance, opseq: list[int]) -> list[list[int]]:
    """Operation-based sequence (job ids with repetition) -> machine sequences (always feasible)."""
    m = inst.m
    nxt = [0] * inst.n
    seq: list[list[int]] = [[] for _ in range(m)]
    for j in opseq:
        o = j * m + nxt[j]
        nxt[j] += 1
        seq[inst.mach[o]].append(o)
    return seq


def identity_start(inst: Instance) -> list[list[int]]:
    """Job shop analog of the identity permutation: jobs dispatched round-robin by index."""
    return decode_opseq(inst, [j for _ in range(inst.m) for j in range(inst.n)])


def random_start(inst: Instance, rng: random.Random) -> list[list[int]]:
    ops = [j for j in range(inst.n) for _ in range(inst.m)]
    rng.shuffle(ops)
    return decode_opseq(inst, ops)


def topo_opseq(inst: Instance, seq) -> list[int]:
    """Machine sequences -> operation-based sequence (topological order by heads)."""
    ev = evaluate(inst, seq, need_tails=False)
    order = sorted(range(inst.N), key=lambda o: (ev.r[o], o))
    return [o // inst.m for o in order]


def double_bridge(inst: Instance, seq, rng: random.Random) -> list[list[int]]:
    """Double bridge (A|B|C|D -> A|C|B|D) on the operation-based sequence + decode."""
    ops = topo_opseq(inst, seq)
    if len(ops) < 4:
        return [s[:] for s in seq]
    a, b, c = sorted(rng.sample(range(1, len(ops)), 3))
    return decode_opseq(inst, ops[:a] + ops[b:c] + ops[a:b] + ops[c:])


def seq_key(seq) -> tuple:
    return tuple(tuple(s) for s in seq)
