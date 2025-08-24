import random

from src.decoder import create_random_permutation
from src.operations import create_base_permutation, validate_permutation
from src.parser import parse_taillard_data
from src.search import evaluate, hill_climb, simulated_annealing, tabu_search
from src.search import CacheType


def test_evaluate_cache_reuses_schedule() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    perm = create_base_permutation(inst)
    cache: CacheType = {}
    c1, sched1 = evaluate(
        inst,
        perm,
        cache=cache,
        return_schedule=True,
    )
    c2, sched2 = evaluate(
        inst,
        perm,
        cache=cache,
        return_schedule=True,
    )
    assert c1 == c2
    assert sched1 is sched2  # object reused from cache
    assert len(cache) == 1


def test_hill_climb_not_worse_best_improvement() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    cache: CacheType = {}
    start_c = evaluate(inst, start, cache=cache)
    best_perm, best_c, iters, evals = hill_climb(
        inst,
        start,
        neighbor_limit=20,
        max_no_improve=5,
        best_improvement=True,
        rng=random.Random(0),
        cache=cache,
    )
    validate_permutation(inst, best_perm)
    assert best_c <= start_c
    assert iters > 0
    assert evals >= iters  # at least one evaluation per iteration expected


def test_hill_climb_first_improvement_runs() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    rng = random.Random(1)
    start = create_random_permutation(inst, rng=rng)
    start_c = evaluate(inst, start)
    best_perm, best_c, iters, evals = hill_climb(
        inst,
        start,
        neighbor_limit=15,
        max_no_improve=5,
        best_improvement=False,
        rng=rng,
    )
    validate_permutation(inst, best_perm)
    assert best_c <= start_c
    assert iters > 0
    assert evals >= iters


def test_tabu_search_runs_and_not_worse() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    start_c = evaluate(inst, start)
    best_perm, best_c, evals = tabu_search(
        inst,
        start,
        iterations=50,
        tenure=10,
        candidate_size=40,
        rng=random.Random(0),
    )
    validate_permutation(inst, best_perm)
    assert best_c <= start_c
    assert evals > 0


def test_hill_climb_best_vs_first() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    start_c = evaluate(inst, start)
    perm_best, c_best, _, _ = hill_climb(
        inst,
        start,
        neighbor_limit=25,
        max_no_improve=5,
        best_improvement=True,
        rng=random.Random(1),
    )
    perm_first, c_first, _, _ = hill_climb(
        inst,
        start,
        neighbor_limit=25,
        max_no_improve=5,
        best_improvement=False,
        rng=random.Random(1),
    )
    validate_permutation(inst, perm_best)
    validate_permutation(inst, perm_first)
    assert c_best <= start_c
    assert c_first <= start_c
    # Strategies can yield different results; both should improve start.
    # No deterministic guarantee best-improvement <= first-improvement under
    # identical seed with a limited sampled neighborhood.


def test_tabu_small_candidate_size() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    start_c = evaluate(inst, start)
    best_perm, best_c, evals = tabu_search(
        inst,
        start,
        iterations=40,
        tenure=8,
        candidate_size=5,
        rng=random.Random(2),
    )
    validate_permutation(inst, best_perm)
    assert evals > 0
    assert best_c <= start_c


def test_simulated_annealing_runs_and_not_worse() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    start_c = evaluate(inst, start)
    best_perm, best_c, evals, iters = simulated_annealing(
        inst,
        start,
        iterations=300,
        initial_temp=25.0,
        cooling=0.97,
        neighbor_moves=2,
        rng=random.Random(3),
    )
    validate_permutation(inst, best_perm)
    assert best_c <= start_c
    assert evals > 0
    assert iters > 0


def test_simulated_annealing_min_temp_stops_early() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    start = create_base_permutation(inst)
    # Set high min_temp so it stops after exactly one iteration
    best_perm, best_c, evals, iters = simulated_annealing(
        inst,
        start,
        iterations=100,
        initial_temp=1.0,
        cooling=0.5,  # po 1 iteracji T=0.5 < min_temp => stop
        neighbor_moves=1,
        rng=random.Random(4),
        min_temp=0.9,
    )
    validate_permutation(inst, best_perm)
    assert iters == 1
    assert evals >= 1
