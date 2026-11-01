"""T2 check on full block shuffles: plain (loose) vs strict E-path predicate.

Usage: python scripts/jobshop_t2_strict_check.py <seed> <reps>   (pilot: seeds 1-3, reps 150)
"""
import sys, random
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import jobshop_proto as J

def reach_avoid(inst, ev, src, dst, forbid):
    if src == dst:
        return True
    if src in forbid:
        return False
    m, r = inst.m, ev.r
    lim = r[dst]
    stack, seen = [src], {src}
    while stack:
        u = stack.pop()
        for v in ((u + 1) if (u + 1) % m else -1, ev.msucc[u]):
            if v < 0 or v in seen or r[v] > lim or (v in forbid and v != dst):
                continue
            if v == dst:
                return True
            seen.add(v); stack.append(v)
    return False

def pred(inst, ev, path, blks, orders, strict):
    m = inst.m
    for (s, L), new in zip(blks, orders):
        old = path[s:s + L]
        if new == old:
            continue
        forbid = set(old)
        npos = {o: t for t, o in enumerate(new)}
        for a in range(L):
            for b in range(a + 1, L):
                x, y = old[a], old[b]
                if npos[y] > npos[x]:
                    continue
                if (x + 1) % m == 0 or y % m == 0:
                    continue
                js, jp = x + 1, y - 1
                ok = reach_avoid(inst, ev, js, jp, forbid) if strict else J.reach(inst, ev, js, jp)
                if ok:
                    return True
    return False

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
cnt = dict(tested=0, infeasible=0, mis_loose=0, mis_strict=0, loose_over=0)
for rep in range(int(sys.argv[2]) if len(sys.argv) > 2 else 150):
    n, m = [(8, 4), (10, 5), (15, 5), (12, 6), (20, 5), (15, 10)][rep % 6]
    inst = J.random_instance(rng, n, m, [1.0, 0.5, 0.25][rep % 3])
    for kind, seq in J.states_for(inst, rng, 2, 2):
        ev = J.evaluate(inst, seq)
        path = J.critical_path(inst, ev)
        blks = J.blocks_of(path, ev)
        big = [b for b, (s, L) in enumerate(blks) if L >= 4]
        if not big:
            continue
        for trial in range(40):
            orders = [path[s:s + L] for s, L in blks]
            for b in rng.sample(big, min(len(big), rng.choice([1, 2]))):
                seg = orders[b][:]
                rng.shuffle(seg)          # full-block shuffle: many reversals at once
                orders[b] = seg
            real = J.cmax(inst, J.apply_block_orders(inst, seq, ev, path, blks, orders)) is None
            pl = pred(inst, ev, path, blks, orders, strict=False)
            ps = pred(inst, ev, path, blks, orders, strict=True)
            cnt['tested'] += 1; cnt['infeasible'] += real
            cnt['mis_loose'] += (pl != real); cnt['mis_strict'] += (ps != real)
            cnt['loose_over'] += (pl and not real)
print(cnt)
