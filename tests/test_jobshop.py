"""Tests of the job shop layer (src/jobshop). No network, no data files: instances are generated
with Taillard's generator (and the stored seeds of ta01)."""

import itertools
import random

import pytest

from src.jobshop import instance as I
from src.jobshop.algorithms import iterated_local_search, simulated_annealing
from src.jobshop.feasibility import blocks_reorder_feasible, order_feasible
from src.jobshop.neighborhoods import (apply_moves, best_move, candidates, estimate, new_order,
                                       parse_mode)
from src.jobshop.runner import JSRunConfig, gen_seeds, load_instance, run_one
from src.jobshop.schedule import (blocks_of, cmax, critical_path, evaluate, identity_start,
                                  random_start)

MODES = ["js_n1", "js_n5", "js_fib", "js_k3", "js_k4", "js_kn", "js_n6", "js_n7", "js_c6", "js_c7",
         "js_bm3", "js_bmn", "js_k3_exact", "js_c6_job"]


def _inst(n, m, rho=1.0, k=0):
    ts, ms = gen_seeds(n, m, k)
    return I.taillard_generate(ts, ms, n, m, rho)


def _states(inst, rng, nrand=2, nlocal=2):
    out = [random_start(inst, rng) for _ in range(nrand)]
    spec = parse_mode("js_n1")
    for _ in range(nlocal):
        seq = random_start(inst, rng)
        for _ in range(200):
            new, c, _ = best_move(inst, seq, spec)
            if new is None or c >= cmax(inst, seq):
                break
            seq = new
        out.append(seq)
    return out


def test_ta01_generator_from_stored_seeds():
    e = I.taillard_table()["ta01"]
    g = I.taillard_generate(e["time_seed"], e["machine_seed"], e["n"], e["m"])
    assert g.p[0][:5] == [94, 66, 10, 53, 26]
    assert [x + 1 for x in g.route[0][:5]] == [7, 13, 5, 8, 4]
    assert len(I.taillard_table()) == 80


def test_rho_zero_is_flow_shop_and_times_do_not_depend_on_rho():
    a, b = _inst(8, 4, 0.0), _inst(8, 4, 0.7)
    assert all(r == list(range(4)) for r in a.route)
    assert a.p == b.p


def test_starts_feasible_and_lb():
    inst = _inst(10, 5)
    for seq in (identity_start(inst), random_start(inst, random.Random(1))):
        c = cmax(inst, seq)
        assert c is not None and c >= inst.lb


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_runs(mode):
    inst = _inst(10, 5, k=3)
    seq = identity_start(inst)
    new, c, mid = best_move(inst, seq, parse_mode(mode))
    assert new is not None and c == cmax(inst, new) and mid


def test_theorem_T2_on_random_block_permutations():
    rng = random.Random(7)
    tested = 0
    for rep in range(30):
        n, m = [(6, 4), (10, 5), (12, 6)][rep % 3]
        inst = _inst(n, m, [1.0, 0.5, 0.25][rep % 3], k=rep)
        for seq in _states(inst, rng):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev)
            blks = blocks_of(path, ev)
            big = [b for b, (s, L) in enumerate(blks) if L >= 3]
            for _ in range(8 if big else 0):
                orders = [path[s:s + L] for s, L in blks]
                b = rng.choice(big)
                seg = orders[b][:]
                rng.shuffle(seg)
                orders[b] = seg
                new = [x[:] for x in seq]
                for (s, L), order in zip(blks, orders):
                    k, p0 = inst.mach[path[s]], ev.pos[path[s]]
                    new[k][p0:p0 + L] = order
                real = cmax(inst, new) is not None
                assert real == blocks_reorder_feasible(inst, ev, path, blks, orders)
                tested += 1
    assert tested > 100


def test_adjacent_critical_swaps_feasible_and_estimate_is_lower_bound():
    rng = random.Random(3)
    for rep in range(20):
        inst = _inst(8, 4, 1.0, k=100 + rep)
        for seq in _states(inst, rng):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev)
            for s, L in blocks_of(path, ev):
                for a in range(L - 1):
                    moves = [(s + a, s + a + 1, "swap")]
                    c = cmax(inst, apply_moves(inst, seq, ev, path, moves))
                    assert c is not None  # T3a
                    k, pa = inst.mach[path[s + a]], ev.pos[path[s + a]]
                    est = estimate(inst, seq, ev, k, pa, pa + 1, new_order(seq[k][pa:pa + 2], "swap"))
                    assert est <= c  # T5
                    if est >= ev.C:
                        assert est == c


def test_lossless_border_filter_single_swaps():
    """Improving swap inside a block must replace a (proper) border operation (T4)."""
    rng = random.Random(11)
    for rep in range(20):
        inst = _inst(8, 4, 1.0, k=200 + rep)
        for seq in _states(inst, rng):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev)
            blks = blocks_of(path, ev)
            nb = len(blks)
            for bidx, (s, L) in enumerate(blks):
                for a, b in itertools.combinations(range(L), 2):
                    c = cmax(inst, apply_moves(inst, seq, ev, path, [(s + a, s + b, "swap")]))
                    if c is not None and c < ev.C:
                        assert nb > 1
                        assert (bidx != 0 and a == 0) or (bidx != nb - 1 and b == L - 1)


def test_disjoint_feasible_border_moves_compose():
    """T3b: disjoint intervals of individually feasible moves give a feasible compound move."""
    rng = random.Random(5)
    spec = parse_mode("js_bmn")
    for rep in range(20):
        inst = _inst(10, 5, 0.5, k=300 + rep)
        for seq in _states(inst, rng):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev)
            feas = []
            for i, j, t in candidates(path, blocks_of(path, ev), spec):
                k = inst.mach[path[i]]
                pa, pb = ev.pos[path[i]], ev.pos[path[j]]
                old = seq[k][pa:pb + 1]
                ok = order_feasible(inst, ev, old, new_order(old, t))
                assert ok == (cmax(inst, apply_moves(inst, seq, ev, path, [(i, j, t)])) is not None)
                if ok:
                    feas.append((i, j, t))
            rng.shuffle(feas)
            chosen = []
            for c in feas:
                if all(c[1] < d[0] or d[1] < c[0] for d in chosen):
                    chosen.append(c)
            assert cmax(inst, apply_moves(inst, seq, ev, path, chosen)) is not None


def test_flow_shop_every_block_reordering_feasible():
    """T8: with identical routes no reordering inside critical blocks creates a cycle."""
    rng = random.Random(2)
    for rep in range(10):
        inst = _inst(10, 4, 0.0, k=400 + rep)
        for seq in _states(inst, rng):
            ev = evaluate(inst, seq)
            path = critical_path(inst, ev)
            for s, L in blocks_of(path, ev):
                if L >= 3:
                    k, p0 = inst.mach[path[s]], ev.pos[path[s]]
                    for _ in range(5):
                        new = [x[:] for x in seq]
                        seg = new[k][p0:p0 + L]
                        rng.shuffle(seg)
                        new[k][p0:p0 + L] = seg
                        assert cmax(inst, new) is not None


@pytest.mark.parametrize("algo", ["ils", "sa"])
def test_algorithms_and_result_format(tmp_path, algo):
    cfg = JSRunConfig(algorithm=algo, neighborhood="js_c6", instance_file="gen:10x5:rho=1.0:0",
                      instance_number=0, seed=0, time_limit_ms=300, tabu_tenure=10 if algo == "ils" else None)
    d = run_one(cfg, tmp_path)
    inst = load_instance(cfg.instance_file)
    assert d["cmax_best"] == cmax(inst, d["best_sequences"])
    assert d["cmax_best"] >= inst.lb and d["iterations"] > 0
    log = d["move_log"]
    assert all(log[i][1] >= log[i + 1][1] for i in range(len(log) - 1))  # best C is monotone
    assert log[-1][1] == d["cmax_best"]
    assert list(tmp_path.rglob("result.json"))


def test_ils_and_sa_direct_calls():
    inst = _inst(8, 4, 1.0, k=9)
    for fn in (iterated_local_search, simulated_annealing):
        r = fn(inst, "js_k3", 150, 0)
        assert r["cmax_best"] <= cmax(inst, identity_start(inst))
