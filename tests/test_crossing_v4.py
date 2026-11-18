"""Plan v4 additions to the crossing family: _k (equal cardinality), _top (exact acceptance among several
moves), _lim (caps), cross_q<K> (pair-corrected QUBO), NEH start and IG (articles/crossings/PLAN.md, note v4)."""

import itertools
import random

import pytest

from src.algorithms.ils import iterated_local_search
from src.algorithms.neh_ig import best_insertion, iterated_greedy, neh
from src.neighborhoods.classical import crossing as cr
from src.neighborhoods.classical.kbonacci import compute_candidates
from src.permutation_procesing import c_max


def rand_instance(n, m, seed):
    rng = random.Random(seed)
    return [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]


def states(count=12, n=14, m=5):
    for s in range(count):
        p = rand_instance(n, m, s)
        pi = list(range(n))
        random.Random(100 + s).shuffle(pi)
        yield p, pi


def test_parse_options():
    o = cr.parse_crossing_options("cross_inv_k4_cp")
    assert (o["family"], o["card"], o["boundary"], o["top"]) == ("inv", 4, "critical_path", None)
    o = cr.parse_crossing_options("cross_c0_top8")
    assert (o["family"], o["top"], o["rule"]) == ("c0", 8, "plain")
    o = cr.parse_crossing_options("cross_inv_lim10000x2000")
    assert (o["top_k"], o["core_k"]) == (10000, 2000)
    o = cr.parse_crossing_options("cross_q16")
    assert (o["family"], o["q"]) == ("q", 16)
    assert cr.parse_crossing_mode("cross_c0_safe_cp") == ("c0", "safe", "critical_path")
    assert cr.parse_crossing_options("motzkin") is None
    for bad in ("cross_q40", "cross_q16_safe", "cross_swap_k2", "cross_c0_top8_safe", "cross_c0_foo", "cross_xx"):
        with pytest.raises(ValueError):
            cr.parse_crossing_options(bad)


@pytest.mark.parametrize("family", ["dyn", "c0", "pa1", "inv"])
def test_card_cuts_to_most_improving_swaps(family):
    for p, pi in states():
        st0, st = {}, {}
        _, _, full = cr.crossing_neighborhood(pi, p, family, stats=st0)
        _, _, cut = cr.crossing_neighborhood(pi, p, family, stats=st, card=2)
        assert len(cut) <= 2 and set(cut) <= set(full)
        if len(full) > 2:   # the two most improving arcs of the full move
            _, cand = compute_candidates(pi, p, len(pi), True, "last_machine")
            d = {(i, j): x for i, j, x in cand}
            assert sorted(cut) == sorted(sorted(full, key=lambda a: d[a])[:2])


@pytest.mark.parametrize("family", ["c0", "inv"])
def test_top1_equals_safe(family):
    for p, pi in states():
        a = cr.crossing_neighborhood(pi, p, family, rule="safe")
        b = cr.crossing_neighborhood(pi, p, family, top=1)
        assert a[1] == b[1]


@pytest.mark.parametrize("family", ["dyn", "c0", "pa1", "inv"])
def test_top_never_worse_than_plain_and_single(family):
    for p, pi in states():
        plain = cr.crossing_neighborhood(pi, p, family)
        single = cr.crossing_neighborhood(pi, p, "swap")
        new_pi, new_c, arcs = cr.crossing_neighborhood(pi, p, family, top=8)
        assert new_c <= plain[1] and new_c <= single[1]
        assert new_c == c_max(new_pi, p)


def test_lim_caps_change_nothing_when_loose():
    for p, pi in states():
        for f in ("c0", "pa1", "inv"):
            assert cr.crossing_neighborhood(pi, p, f)[1] == cr.crossing_neighborhood(pi, p, f, top_k=10**6, core_k=10**6)[1]


def brute_qubo(lin, quad, conf):
    K, bad = len(lin), set(conf)
    best = (0.0, [])
    for r in range(1, K + 1):
        for S in itertools.combinations(range(K), r):
            if any((x, y) in bad for x, y in itertools.combinations(S, 2)):
                continue
            e = sum(lin[x] for x in S) + sum(quad.get((x, y), 0.0) for x, y in itertools.combinations(S, 2))
            if e < best[0] - 1e-12:
                best = (e, list(S))
    return best


def test_qubo_exact_matches_brute_force_and_real_pairs():
    for p, pi in states(8, n=12, m=4):
        base, cand = compute_candidates(pi, p, len(pi), True, "last_machine")
        imp = {(i, j): d for i, j, d in cand if d < 0}
        if len(imp) < 3:
            continue
        arcs, lin, quad, conf = cr.pair_corrected_qubo(pi, p, base, imp, 10)
        e, idx = cr.solve_qubo_exact(lin, quad, conf)
        assert abs(e - brute_qubo(lin, quad, conf)[0]) < 1e-9
        for (x, y), w in quad.items():   # a pair's energy is its real joint gain
            q = cr._apply(pi, [arcs[x], arcs[y]])
            assert abs(lin[x] + lin[y] + w - (c_max(q, p) - base)) < 1e-9
        e0, idx0 = cr.solve_qubo_exact(lin, {}, conf)   # no couplers: maximum-weight matching on the same arcs
        assert abs(e0 - cr.select_inv({arcs[x]: lin[x] for x in range(len(arcs))})[0]) < 1e-9


def test_q_mode_returns_consistent_move():
    for p, pi in states(5):
        new_pi, new_c, arcs = cr.crossing_neighborhood(pi, p, "q", q=8)
        assert new_c == c_max(new_pi, p) and len({x for a in arcs for x in a}) == 2 * len(arcs)


def test_neh_and_insertion():
    p = rand_instance(9, 4, 1)
    seq = [3, 1, 4]
    t, c = best_insertion(seq, 7, p)
    brute = min(range(4), key=lambda t: (c_max(seq[:t] + [7] + seq[t:], p), t))
    assert t == brute and c == c_max(seq[:t] + [7] + seq[t:], p)
    s = neh(p)
    assert sorted(s) == list(range(9)) and c_max(s, p) <= c_max(list(range(9)), p) * 1.5


def test_neh_taillard_ta001():
    from src.parser import parser
    p = parser("data/tai20_5.txt", 0)["processing_times"]
    assert c_max(neh(p), p) == 1286   # NEH value of ta001 reported in the literature


def test_ig_and_ils_neh_run():
    p = rand_instance(15, 5, 3)
    random.seed(0)
    best, c, *_ = iterated_greedy(p, max_time_ms=300)
    assert sorted(best) == list(range(15)) and c == c_max(best, p) <= c_max(neh(p), p)
    random.seed(0)
    best, c, *_ = iterated_local_search(p, max_time_ms=300, neigh_mode="cross_c0_top4", start="neh")
    assert c == c_max(best, p) <= c_max(neh(p), p)
    for mode in ("cross_inv_k2", "cross_q8", "cross_inv_lim50x10"):
        random.seed(0)
        best, c, *_ = iterated_local_search(p, max_time_ms=200, neigh_mode=mode)
        assert c == c_max(best, p)
