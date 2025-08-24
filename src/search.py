"""Search utilities: evaluation caching, hill climbing and tabu search.

Overview
--------
This module provides lightweight metaheuristic building blocks around the
decoder. Emphasis is on clarity and testability rather than algorithmic
exhaustiveness.

Caching strategy
----------------
``evaluate`` memoises decoded schedules keyed by the tuple form of a
permutation (allowing callers to keep list objects mutable). Each entry maps
to ``(cmax, Schedule)`` enabling reuse of the full schedule when needed.

Local search
------------
``hill_climb`` explores a sampled neighborhood (see ``generate_neighbors``)
with either best- or first-improvement selection, terminating after a number
of consecutive non-improving iterations.

Tabu search
-----------
``tabu_search`` applies swap-based moves with a fixed tenure tabu list and
optional aspiration criterion that allows overriding tabu status when a move
improves the global best.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .decoder import build_schedule_from_permutation
from .models import DataInstance, OperationKey, Schedule
from .neighborhood import generate_neighbors, swap_any

CacheType = Dict[Tuple[OperationKey, ...], Tuple[int, Schedule]]


def evaluate(
    data: DataInstance,
    permutation: Iterable[OperationKey],
    cache: Optional[CacheType] = None,
    return_schedule: bool = False,
    validate: bool = False,
) -> int | Tuple[int, Schedule]:
    """Decode a permutation and return its makespan.

    Args:
        data: Problem instance.
        permutation: Iterable of operation keys forming a feasible (job-order
            respecting) permutation.
        cache: Optional mutable dict used for memoisation. If provided and an
            entry exists, decoding is skipped.
        return_schedule: When True also return the full ``Schedule`` object.
        validate: When True the decoder performs additional sanity checks
            (slower; intended for tests / debugging).

    Returns:
        Either the makespan (int) or a tuple ``(makespan, Schedule)`` when
        ``return_schedule`` is True.
    """
    key = tuple(permutation)
    if cache is not None and key in cache:
        cmax, sched = cache[key]
        return (cmax, sched) if return_schedule else cmax
    sched = build_schedule_from_permutation(
        data,
        key,  # tuple is iterable of OperationKey
        validate=validate,
        check_completeness=True,
    )
    if cache is not None:
        cache[key] = (sched.cmax, sched)
    return (sched.cmax, sched) if return_schedule else sched.cmax


def hill_climb(
    data: DataInstance,
    start_perm: List[OperationKey],
    neighbor_limit: int = 50,
    max_no_improve: int = 50,
    best_improvement: bool = True,
    rng: Optional[random.Random] = None,
    cache: Optional[CacheType] = None,
    progress: Optional[List[int]] = None,
    time_progress: Optional[List[float]] = None,
) -> Tuple[List[OperationKey], int, int, int]:
    """Perform hill climbing from a starting permutation.

    Args:
        data: Problem instance.
        start_perm: Initial feasible permutation.
        neighbor_limit: Number of neighbors sampled per iteration.
        max_no_improve: Termination criterion: stop after this many
            consecutive iterations without improvement of the current solution.
        best_improvement: If True evaluate all sampled neighbors and move to
            the best improving one; if False accept the first improving
            neighbor (after shuffling) for potentially faster progress.
        rng: Optional random generator for reproducibility.
        cache: Shared evaluation cache (allows synergy with multi-start /
            tabu).

    Returns:
        Tuple ``(best_perm, best_cmax, iterations, evaluations)`` where:
            best_perm: Best permutation discovered (list copy).
            best_cmax: Its makespan.
            iterations: Number of loop iterations executed.
            evaluations: Number of (unique or cached) evaluate calls counted.
    """
    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}

    current = list(start_perm)
    current_cmax = evaluate(
        data,
        current,
        cache=cache,
    )  # type: ignore[arg-type]
    best_perm = list(current)
    best_cmax = current_cmax  # type: ignore[assignment]
    iters = 0
    evals = 1
    no_imp = 0

    t0 = time.perf_counter()
    if progress is not None:
        progress.append(best_cmax)  # initial state
    if time_progress is not None:
        time_progress.append(0.0)
    while no_imp < max_no_improve:
        iters += 1
        improved = False
        neighbors = generate_neighbors(current, neighbor_limit, rng=rng)
        if best_improvement:
            candidate_best = None
            candidate_perm = None
            for n in neighbors:
                c = evaluate(data, n, cache=cache)  # type: ignore[arg-type]
                evals += 1
                if candidate_best is None or c < candidate_best:
                    candidate_best = c
                    candidate_perm = n
            if (
                candidate_best is not None
                and candidate_best < current_cmax  # type: ignore[operator]
            ):
                current = candidate_perm  # type: ignore[assignment]
                current_cmax = candidate_best  # type: ignore[assignment]
                improved = True
        else:  # first improvement
            rng.shuffle(neighbors)
            for n in neighbors:
                c = evaluate(data, n, cache=cache)  # type: ignore[arg-type]
                evals += 1
                if c < current_cmax:  # type: ignore[operator]
                    current = n
                    current_cmax = c  # type: ignore[assignment]
                    improved = True
                    break
        if improved:
            if current_cmax < best_cmax:  # type: ignore[operator]
                best_cmax = current_cmax  # type: ignore[assignment]
                best_perm = list(current)
            no_imp = 0
        else:
            no_imp += 1
        if progress is not None:
            progress.append(best_cmax)  # type: ignore[arg-type]
        if time_progress is not None:
            time_progress.append(time.perf_counter() - t0)
    return best_perm, best_cmax, iters, evals


def tabu_search(
    data: DataInstance,
    start_perm: List[OperationKey],
    iterations: int = 200,
    tenure: int = 15,
    candidate_size: int = 60,
    rng: Optional[random.Random] = None,
    cache: Optional[CacheType] = None,
    aspiration: bool = True,
    progress: Optional[List[int]] = None,
    time_progress: Optional[List[float]] = None,
) -> Tuple[List[OperationKey], int, int]:
    """Run a simple swap-based Tabu Search.

    Args:
        data: Problem instance.
        start_perm: Starting feasible permutation.
        iterations: Maximum number of main iterations.
        tenure: Number of iterations a performed move remains tabu.
        candidate_size: Number of distinct swap index pairs sampled each
            iteration (upper bound on neighborhood size considered).
        rng: Optional random generator.
        cache: Shared evaluation cache.
        aspiration: If True allow a tabu move when it produces a new global
            best solution.

    Returns:
        Tuple ``(best_perm, best_cmax, evaluations)`` where evaluations counts
        total calls to ``evaluate`` (cache hits included in the count for
        comparability across runs).
    """
    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}
    current = list(start_perm)
    current_c = evaluate(data, current, cache=cache)  # type: ignore[arg-type]
    best_perm = list(current)
    best_c = current_c  # type: ignore[assignment]
    evals = 1

    # tabu dict: move_descriptor -> expire_iteration
    tabu: dict[Tuple[OperationKey, OperationKey], int] = {}

    n = len(current)
    t0 = time.perf_counter()
    if progress is not None:
        progress.append(best_c)
    if time_progress is not None:
        time_progress.append(0.0)
    for it in range(1, iterations + 1):
        candidates: list[Tuple[int, int]] = []
        seen_pairs: set[Tuple[int, int]] = set()
        # sample candidate_size unique index pairs
        attempts = 0
        while (
            len(candidates) < candidate_size
            and attempts < candidate_size * 5
        ):
            attempts += 1
            i, j = sorted(rng.sample(range(n), 2))
            if (i, j) in seen_pairs:
                continue
            if current[i][0] == current[j][0]:  # same job -> skip
                continue
            seen_pairs.add((i, j))
            candidates.append((i, j))
        if not candidates:
            break

        best_move = None
        best_move_perm = None
        best_move_c = None

        for i, j in candidates:
            op_i, op_j = current[i], current[j]
            move_key = tuple(sorted((op_i, op_j)))  # type: ignore[arg-type]
            new_perm = swap_any(current, i, j)
            if new_perm == current:
                continue
            c = evaluate(data, new_perm, cache=cache)  # type: ignore[arg-type]
            evals += 1
            is_tabu = move_key in tabu and tabu[move_key] >= it
            if is_tabu and not (aspiration and c < best_c):
                continue
            if best_move_c is None or c < best_move_c:
                best_move_c = c
                best_move = move_key
                best_move_perm = new_perm

        if best_move_perm is None:
            # all moves tabu & no aspiration; decay tabu and continue
            # Optionally we could clear, but here just proceed.
            continue

        # apply best move
        current = best_move_perm
        current_c = best_move_c  # type: ignore[assignment]
        # register tabu
        if best_move is not None:
            tabu[best_move] = it + tenure
        # cleanup expired entries
        expired = [k for k, exp in tabu.items() if exp < it]
        for k in expired:
            del tabu[k]
        # update global best
        if current_c < best_c:  # type: ignore[operator]
            best_c = current_c  # type: ignore[assignment]
            best_perm = list(current)
        if progress is not None:
            progress.append(best_c)  # type: ignore[arg-type]
        if time_progress is not None:
            time_progress.append(time.perf_counter() - t0)

    return best_perm, best_c, evals


def simulated_annealing(
    data: DataInstance,
    start_perm: List[OperationKey],
    iterations: int = 1000,
    initial_temp: float = 50.0,
    cooling: float = 0.95,
    neighbor_moves: int = 1,
    rng: Optional[random.Random] = None,
    cache: Optional[CacheType] = None,
    min_temp: float = 1e-3,
    progress: Optional[List[int]] = None,
    time_progress: Optional[List[float]] = None,
) -> Tuple[List[OperationKey], int, int, int]:
    """Run Simulated Annealing (SA) on a permutation.

    A random swap (between operations of different jobs) or insertion move is
    sampled to generate a neighbor. Acceptance probability for worsening move
    ``delta = new_c - current_c`` is ``exp(-delta / T)`` with temperature ``T``
    following geometric cooling ``T <- cooling * T`` each iteration.

    Args:
        data: Problem instance.
        start_perm: Initial feasible permutation.
        iterations: Maximum number of SA iterations.
        initial_temp: Starting temperature ``T0``.
        cooling: Multiplicative cooling factor in (0,1).
        neighbor_moves: Number of neighbor trials per iteration (best of trials
            considered for acceptance). Keep small for speed.
        rng: Optional random generator.
        cache: Shared evaluation cache for reuse with other searches.
        min_temp: Early stop if temperature drops below this threshold.

    Returns:
        Tuple ``(best_perm, best_cmax, evaluations, performed_iterations)``.
    """
    import math

    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}
    current = list(start_perm)
    current_c = evaluate(data, current, cache=cache)  # type: ignore[arg-type]
    best_perm = list(current)
    best_c = current_c  # type: ignore[assignment]
    T = float(initial_temp)
    evals = 1
    n = len(current)

    def random_neighbor(base: List[OperationKey]) -> List[OperationKey]:
        # Attempt a feasible random move (swap or insertion)
        for _ in range(10):
            move_type = rng.choice(("swap", "ins"))
            i, j = rng.sample(range(n), 2)
            if move_type == "swap":
                cand = swap_any(base, i, j)
            else:
                # simple insertion using slicing (reuse neighborhood.insertion
                # would import extra symbol; light inline variant) but keep
                # order feasibility by fallback if no change
                from .neighborhood import insertion as _ins

                cand = _ins(base, i, j)
            if cand != base:
                return cand
        return base[:]  # fallback (rare)

    t0 = time.perf_counter()
    if progress is not None:
        progress.append(best_c)
    if time_progress is not None:
        time_progress.append(0.0)
    it = 0
    while it < iterations and T > min_temp:
        it += 1
        candidate_best_perm = None
        candidate_best_c = None
        for _ in range(neighbor_moves):
            neigh = random_neighbor(current)
            c = evaluate(data, neigh, cache=cache)  # type: ignore[arg-type]
            evals += 1
            if candidate_best_c is None or c < candidate_best_c:
                candidate_best_c = c
                candidate_best_perm = neigh
        if candidate_best_perm is None:
            T *= cooling
            continue
        delta = candidate_best_c - current_c  # type: ignore[operator]
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta / T)  # type: ignore[arg-type]
                if rng.random() < prob:
                    accept = True
        if accept:
            current = candidate_best_perm
            current_c = candidate_best_c  # type: ignore[assignment]
            if current_c < best_c:  # type: ignore[operator]
                best_c = current_c  # type: ignore[assignment]
                best_perm = list(current)
        T *= cooling
        if progress is not None:
            progress.append(best_c)  # type: ignore[arg-type]
        if time_progress is not None:
            time_progress.append(time.perf_counter() - t0)
    return best_perm, best_c, evals, it
