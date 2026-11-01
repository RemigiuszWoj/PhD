"""Border-move family on the critical path (articles/jobshop/PLAN.md §2.3-2.4).

A candidate is a move inside one critical block that replaces the block's first or last
operation (lossless border rule, T4; first block: only the last op counts, last block:
only the first). Move types on a block interval [a, b] (block-local, a < b):
    swap   : endpoint swap
    front  : the op at b jumps to a   (left border: N6 "move before the block")
    inward : the op at a moves to b   (right border: N6 "move after the block")
The left border uses intervals [0, d], the right border [L-1-d, L-1], 1 <= d <= k-1.
For d = 1 all types are the same adjacent swap.

Modes (prefix js_):
    js_n1            best single adjacent critical swap (no border filter; van Laarhoven et al.)
    js_n5            best single swap of the first/last pair of a block (Nowicki-Smutnicki)
    js_fib           compound adjacent border swaps (= compound N5, k = 2)
    js_k<K>, js_kn   compound endpoint border swaps of reach <= K-1 (k-bonacci), any reach (Dynasearch)
    js_n6, js_n7     best single insertion (N6: ops jump to the block ends; N7: + end ops move inward)
    js_c6, js_c7     compound N6 / N7 (one move per block end, disjoint intervals, DP)
    js_bm<K>, js_bmn compound border moves of all types (swap + N6 + N7), reach <= K-1 / any
Suffixes (ablations): _exact (exact delta per candidate instead of the block estimate),
    _job (critical path ties go to the job predecessor).

Selection: estimate (or exact) delta per feasible candidate; single modes take the best,
compound modes run the positional DP of the PFSP code over path positions (it decomposes
into independent block optima, T7) and fall back to the best single candidate when the DP
selects nothing (as force_move_if_none in the PFSP k-bonacci). The chosen move is
evaluated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.jobshop.feasibility import order_feasible
from src.jobshop.instance import Instance
from src.jobshop.schedule import Eval, blocks_of, critical_path, evaluate

INF = 10**9
_N6 = (("L", "front"), ("R", "inward"))
_N7 = _N6 + (("L", "inward"), ("R", "front"))
_SW = (("L", "swap"), ("R", "swap"))
_ALL = _N7 + _SW


@dataclass(frozen=True)
class ModeSpec:
    name: str
    kmax: int
    combos: tuple
    compound: bool
    border_filter: bool = True
    exact: bool = False
    prefer: str = "machine"


def parse_mode(mode: str) -> ModeSpec:
    base, exact, prefer = mode, False, "machine"
    for suf in ("_exact", "_job"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            exact = exact or suf == "_exact"
            prefer = "job" if suf == "_job" else prefer
    if not base.startswith("js_"):
        raise ValueError(f"not a job shop mode: {mode!r}")
    b = base[3:]
    table = {
        "n1": (2, _SW, False, False), "n5": (2, _SW, False, True), "fib": (2, _SW, True, True),
        "kn": (INF, _SW, True, True), "n6": (INF, _N6, False, True), "n7": (INF, _N7, False, True),
        "c6": (INF, _N6, True, True), "c7": (INF, _N7, True, True), "bmn": (INF, _ALL, True, True),
    }
    if b in table:
        k, c, comp, filt = table[b]
    elif b.startswith("k") and b[1:].isdigit() and int(b[1:]) >= 2:
        k, c, comp, filt = int(b[1:]), _SW, True, True
    elif b.startswith("bm") and b[2:].isdigit() and int(b[2:]) >= 2:
        k, c, comp, filt = int(b[2:]), _ALL, True, True
    else:
        raise ValueError(f"unknown job shop mode: {mode!r}")
    return ModeSpec(mode, k, c, comp, filt, exact, prefer)


def new_order(seg: list[int], typ: str) -> list[int]:
    if typ == "swap":
        return [seg[-1]] + seg[1:-1] + [seg[0]]
    if typ == "front":
        return [seg[-1]] + seg[:-1]
    if typ == "inward":
        return seg[1:] + [seg[0]]
    raise ValueError(typ)


def candidates(path, blks, spec: ModeSpec) -> list[tuple[int, int, str]]:
    """(i, j, typ) in path positions; the interval [i, j] lies inside one block."""
    out = set()
    nb = len(blks)
    for bidx, (s, L) in enumerate(blks):
        if L < 2:
            continue
        if not spec.border_filter:  # js_n1: every adjacent critical pair
            out.update((s + a, s + a + 1, "swap") for a in range(L - 1))
            continue
        if nb == 1:
            continue  # a single block: C_max = machine load, nothing can improve
        for d in range(1, min(spec.kmax - 1, L - 1) + 1):
            for side, typ in spec.combos:
                t = "swap" if d == 1 else typ
                if side == "L" and bidx != 0:
                    out.add((s, s + d, t))
                if side == "R" and bidx != nb - 1:
                    out.add((s + L - 1 - d, s + L - 1, t))
    return sorted(out)


def estimate(inst: Instance, seq, ev: Eval, k: int, pa: int, pb: int, new: list[int]) -> int:
    """Block-local estimate of C_max after reordering machine k's positions pa..pb as `new`.

    Heads are recomputed left to right and tails right to left inside the interval, with
    the heads of job predecessors / the machine predecessor of the interval and the tails
    of job successors / the machine successor taken from the current solution (Taillard
    1994; Balas and Vazacopoulos 1998). For d = 1 this is an exact lower bound of the new
    C_max, equal to it when the move does not improve (T5).
    """
    m, dur, r, q = inst.m, inst.dur, ev.r, ev.q
    before = seq[k][pa - 1] if pa > 0 else -1
    after = seq[k][pb + 1] if pb + 1 < len(seq[k]) else -1
    heads = []
    prev_end = (r[before] + dur[before]) if before >= 0 else 0
    for o in new:
        jp = o - 1 if o % m else -1
        h = prev_end
        if jp >= 0 and r[jp] + dur[jp] > h:
            h = r[jp] + dur[jp]
        heads.append(h)
        prev_end = h + dur[o]
    nxt = (q[after] + dur[after]) if after >= 0 else 0
    best = 0
    for t in range(len(new) - 1, -1, -1):
        o = new[t]
        js = o + 1 if (o + 1) % m else -1
        tail = nxt
        if js >= 0 and q[js] + dur[js] > tail:
            tail = q[js] + dur[js]
        nxt = tail + dur[o]
        if heads[t] + dur[o] + tail > best:
            best = heads[t] + dur[o] + tail
    return best


def apply_moves(inst: Instance, seq, ev: Eval, path, moves) -> list[list[int]]:
    new = [s[:] for s in seq]
    for i, j, typ in moves:
        k = inst.mach[path[i]]
        pa, pb = ev.pos[path[i]], ev.pos[path[j]]
        new[k][pa : pb + 1] = new_order(seq[k][pa : pb + 1], typ)
    return new


def _dp(n: int, scored):
    """Positional DP of the PFSP k-bonacci code (min sum of deltas over disjoint intervals)."""
    ending_at: dict = {}
    for c in scored:
        ending_at.setdefault(c[1], []).append(c)
    best = [0] * (n + 1)
    choice = [None] * (n + 1)
    for p in range(1, n + 1):
        best[p] = best[p - 1]
        for c in ending_at.get(p - 1, ()):
            val = best[c[0]] + c[2]
            if val < best[p]:
                best[p], choice[p] = val, c
    chosen = []
    p = n
    while p > 0:
        c = choice[p]
        if c is None:
            p -= 1
        else:
            chosen.append(c)
            p = c[0]
    chosen.reverse()
    return best[n], chosen


def best_move(inst: Instance, seq, spec: ModeSpec, ev: Eval | None = None, stats: dict | None = None):
    """Best move of the mode. Returns (new_seq, new_C, move_id) or (None, None, None)."""
    if ev is None:
        ev = evaluate(inst, seq)
    path = critical_path(inst, ev, spec.prefer)
    blks = blocks_of(path, ev)
    scored = []
    n_infeasible = 0
    for i, j, typ in candidates(path, blks, spec):
        k = inst.mach[path[i]]
        pa, pb = ev.pos[path[i]], ev.pos[path[j]]
        old = seq[k][pa : pb + 1]
        new = new_order(old, typ)
        if j - i >= 2 and not order_feasible(inst, ev, old, new):
            n_infeasible += 1
            continue
        if spec.exact:
            ev2 = evaluate(inst, apply_moves(inst, seq, ev, path, [(i, j, typ)]), need_tails=False)
            c = ev2.C
        else:
            c = estimate(inst, seq, ev, k, pa, pb, new)
        scored.append((i, j, c - ev.C, typ))
    if not scored:
        return None, None, None
    if spec.compound:
        pred, chosen = _dp(len(path), scored)
        if not chosen:
            chosen = [min(scored, key=lambda x: x[2])]
            pred = chosen[0][2]
    else:
        chosen = [min(scored, key=lambda x: x[2])]
        pred = chosen[0][2]
    moves = [(c[0], c[1], c[3]) for c in chosen]
    new_seq = apply_moves(inst, seq, ev, path, moves)
    new_ev = evaluate(inst, new_seq, need_tails=False)
    if new_ev is None:  # cannot happen (T3b: disjoint intervals of feasible moves); guard anyway
        return None, None, None
    if stats is not None:
        stats.update(path_len=len(path), blocks=len(blks), candidates=len(scored) + n_infeasible,
                     infeasible=n_infeasible, moves=len(moves), pred_delta=pred, real_delta=new_ev.C - ev.C)
    move_id = frozenset((path[i], path[j], typ) for i, j, typ in moves)
    return new_seq, new_ev.C, move_id
