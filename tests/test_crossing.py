"""Tests for the crossing family (src/neighborhoods/classical/crossing.py, articles/crossings/PLAN.md)."""

import itertools
import random

import pytest

from src.algorithms.ils import iterated_local_search
from src.neighborhoods.classical import crossing as cr
from src.neighborhoods.classical.kbonacci import compute_candidates, kbonacci_neighborhood
from src.neighborhoods.classical.motzkin import motzkin_neighborhood_full
from src.permutation_procesing import c_max


def rand_instance(n, m, seed):
    rng = random.Random(seed)
    return [[rng.randint(1, 99) for _ in range(n)] for _ in range(m)]


def all_involutions(n):
    def rec(free, arcs):
        if not free:
            yield list(arcs)
            return
        a, rest = free[0], free[1:]
        yield from rec(rest, arcs)
        for t, b in enumerate(rest):
            arcs.append((a, b))
            yield from rec(rest[:t] + rest[t + 1:], arcs)
            arcs.pop()
    yield from rec(list(range(n)), [])


FEASIBLE = {
    "dyn": lambda S: all(x[1] < y[0] for x, y in zip(sorted(S), sorted(S)[1:])),
    "c0": lambda S: cr.crossings(S) == 0,
    "c1": lambda S: cr.crossings(S) <= 1,
    "pa1": lambda S: cr.max_crossings_per_arc(S) <= 1,
    "inv": lambda S: True,
}
SELECT = {"dyn": cr.select_dyn, "c0": cr.select_c0, "c1": cr.select_c1, "pa1": cr.select_pa1, "inv": cr.select_inv}


def brute(cands, ok):
    best, arcs = 0.0, sorted(cands)
    for r in range(1, len(arcs) + 1):
        for S in itertools.combinations(arcs, r):
            ends = [p for ab in S for p in ab]
            if len(set(ends)) == len(ends) and ok(S):
                best = min(best, sum(cands[a] for a in S))
    return best


# ---------------------------------------------------------------- counting
@pytest.mark.parametrize("n", range(1, 10))
def test_counts_match_brute_force(n):
    invs = list(all_involutions(n))
    assert cr.count_moves(n, "inv") == len(invs)
    assert cr.count_moves(n, "c0") == sum(1 for s in invs if cr.crossings(s) == 0)
    assert cr.count_moves(n, "c1") == sum(1 for s in invs if cr.crossings(s) <= 1)
    assert cr.count_moves(n, "pa1") == sum(1 for s in invs if cr.max_crossings_per_arc(s) <= 1)
    assert cr.count_moves(n, "dyn") == sum(1 for s in invs if FEASIBLE["dyn"](s))


def test_known_values():
    assert [cr.count_moves(n, "c0") for n in range(1, 11)] == [1, 2, 4, 9, 21, 51, 127, 323, 835, 2188]
    assert cr.count_moves(10, "c1") == 5038 and cr.count_moves(10, "pa1") == 5263 and cr.count_moves(10, "inv") == 9496
    assert sum(cr.count_by_crossings(10, 2)) == 7108          # katalog_sasiedztw.md


# ---------------------------------------------------------------- exact selection
@pytest.mark.parametrize("family", list(SELECT))
def test_selection_matches_brute_force(family):
    rng = random.Random(11)
    for _ in range(120):
        n = rng.randint(4, 10)
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        cands = {ab: -rng.randint(1, 20) for ab in rng.sample(pairs, rng.randint(1, min(len(pairs), 10)))}
        v, arcs = SELECT[family](cands)
        ends = [p for ab in arcs for p in ab]
        assert len(set(ends)) == len(ends)
        assert FEASIBLE[family](arcs)
        assert sum(cands[a] for a in arcs) == pytest.approx(v)
        assert v == pytest.approx(brute(cands, FEASIBLE[family]))


def test_core_cap_keeps_feasibility():
    rng = random.Random(5)
    pairs = [(a, b) for a in range(12) for b in range(a + 1, 12)]
    cands = {ab: -rng.randint(1, 30) for ab in rng.sample(pairs, 30)}
    for per_arc in (False, True):
        v_full, _, cap_full = cr._select_h_family(cands, per_arc, core_k=1000)
        v_cap, arcs, cap = cr._select_h_family(cands, per_arc, core_k=4)
        assert cap and not cap_full
        assert v_cap >= v_full - 1e-9                          # restriction cannot beat the optimum
        assert (cr.max_crossings_per_arc(arcs) <= 1) if per_arc else (cr.crossings(arcs) <= 1)


def test_chain_of_optima():
    """dyn <= c0 <= c1 <= pa1 <= inv in predicted gain (families are nested)."""
    rng = random.Random(3)
    for _ in range(60):
        n = rng.randint(5, 12)
        pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
        cands = {ab: -rng.randint(1, 20) for ab in rng.sample(pairs, rng.randint(1, min(len(pairs), 25)))}
        vals = [SELECT[f](cands)[0] for f in ("dyn", "c0", "c1", "pa1", "inv")]
        assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(4))


# ---------------------------------------------------------------- equivalences with the old code
def test_c0_matches_old_motzkin_value():
    for seed in range(12):
        n, m = random.Random(seed).randint(8, 14), random.Random(seed + 99).randint(2, 5)
        p = rand_instance(n, m, seed)
        pi = list(range(n))
        random.Random(seed).shuffle(pi)
        base, cl = compute_candidates(pi, p, n, True, "last_machine")
        delta = {(i, j): d for i, j, d in cl}
        improving = {ab: d for ab, d in delta.items() if d < 0}
        _, _, old = motzkin_neighborhood_full(pi, p)
        old_val = sum(delta[ab] for ab in old) if improving else None
        new_val, _ = cr.select_c0(improving)
        if improving:
            assert new_val == pytest.approx(old_val)


def test_dyn_plain_is_kbonacci_kn():
    for seed in range(12):
        n, m = 12, 4
        p = rand_instance(n, m, seed)
        pi = list(range(n))
        random.Random(seed).shuffle(pi)
        a = kbonacci_neighborhood(pi, p, n)
        b = cr.crossing_neighborhood(pi, p, "dyn", "plain")
        assert a[0] == b[0] and a[1] == b[1]


# ---------------------------------------------------------------- acceptance rules
def test_safe_and_prefix_never_worse_than_best_single_swap():
    for seed in range(15):
        n, m = 14, 5
        p = rand_instance(n, m, seed)
        pi = list(range(n))
        random.Random(seed + 7).shuffle(pi)
        base, cl = compute_candidates(pi, p, n, True, "last_machine")
        single = base + min(d for _i, _j, d in cl)
        for fam in ("c0", "inv", "pa1"):
            plain = cr.crossing_neighborhood(pi, p, fam, "plain")
            safe = cr.crossing_neighborhood(pi, p, fam, "safe")
            pref = cr.crossing_neighborhood(pi, p, fam, "prefix")
            assert safe[1] <= single and pref[1] <= single
            assert safe[1] <= plain[1] and pref[1] <= safe[1]
            for new_pi, new_c, _ in (plain, safe, pref):
                assert c_max(new_pi, p) == new_c and sorted(new_pi) == list(range(n))


def test_swap_mode_is_best_single_swap():
    p = rand_instance(12, 4, 1)
    pi = list(range(12))
    base, cl = compute_candidates(pi, p, 12, True, "last_machine")
    i, j, d = min(cl, key=lambda c: c[2])
    new_pi, new_c, swaps = cr.crossing_neighborhood(pi, p, "swap")
    assert swaps == [(i, j)] and new_c == base + d


# ---------------------------------------------------------------- modes and integration
def test_parse_modes():
    assert cr.parse_crossing_mode("cross_c0") == ("c0", "plain", "last_machine")
    assert cr.parse_crossing_mode("cross_inv_safe_cp") == ("inv", "safe", "critical_path")
    assert cr.parse_crossing_mode("cross_pa1_prefix") == ("pa1", "prefix", "last_machine")
    assert cr.parse_crossing_mode("cross_swap") == ("swap", "plain", "last_machine")
    assert cr.parse_crossing_mode("motzkin") is None
    with pytest.raises(ValueError):
        cr.parse_crossing_mode("cross_c2")
    with pytest.raises(ValueError):
        cr.parse_crossing_mode("cross_swap_safe")


@pytest.mark.parametrize("mode", ["cross_dyn", "cross_dyn_safe", "cross_c0", "cross_c0_safe", "cross_c1_safe",
                                  "cross_pa1", "cross_inv", "cross_inv_safe", "cross_inv_prefix", "cross_swap",
                                  "cross_c0_safe_cp"])
def test_ils_runs_with_mode(mode):
    random.seed(0)
    p = rand_instance(12, 4, 2)
    cr.reset_crossing_stats()
    best_pi, best_c, _, _, st = iterated_local_search(p, max_time_ms=60, tabu_tenure=10, neigh_mode=mode)
    assert sorted(best_pi) == list(range(12)) and best_c == c_max(best_pi, p)
    s = cr.get_crossing_stats()
    assert s["moves"] >= 1 and s["swaps"] >= s["moves"]
    if not mode.endswith(("safe", "prefix", "safe_cp")):
        assert s["fallback"] == 0
