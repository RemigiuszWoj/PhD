"""Shared primitives used by execution modes.

This module isolates the light data container with algorithm hyper-parameters
(`AlgoParams`) and a thin dispatch helper (`run_algorithm`) so that each mode
(`auto`, `benchmark`, CLI single-algo / pipeline) can invoke metaheuristics
uniformly without duplicating timing / progress collection code.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

from src.models import DataInstance
from src.search import hill_climb, tabu_search, simulated_annealing


@dataclass(slots=True)
class AlgoParams:
    """Bundle of all configurable hyper-parameters for the three
    metaheuristics.

    Keeping them in a single dataclass simplifies passing configuration across
    extracted *mode* functions and makes it straightforward to persist / log
    an experiment setup if desired in the future.
    """
    neighbor_limit: int
    max_no_improve: int
    tabu_iterations: int
    tabu_tenure: int
    tabu_candidate_size: int
    sa_iterations: int
    sa_initial_temp: float
    sa_cooling: float
    sa_neighbor_moves: int


def run_algorithm(
    name: str,
    instance: DataInstance,
    start_perm: list[tuple[int, int]],
    params: AlgoParams,
    rng: random.Random,
    progress: list[int] | None = None,
    time_progress: list[float] | None = None,
) -> tuple[list[tuple[int, int]], int, float]:
    """Execute selected algorithm and return its result triple.

    Args:
        name: One of ``{"hill", "tabu", "sa"}``.
        instance: Parsed problem instance.
        start_perm: Feasible starting permutation.
        params: Hyper-parameter bundle (only the relevant subset is read).
        rng: Random generator forwarded to metaheuristics.
        progress: Optional list mutated in-place with best-so-far Cmax per
            iteration (if algorithm supports reporting).
        time_progress: Optional list mutated with elapsed wall times (s).

    Returns:
        Tuple ``(perm, cmax, elapsed_seconds)`` where ``perm`` is the final
        permutation returned by the algorithm (best found), ``cmax`` its
        objective value and ``elapsed_seconds`` the measured runtime.

    Raises:
        ValueError: If an unknown algorithm name is provided.
    """
    if progress is None:
        progress = []
    if time_progress is None:
        time_progress = []
    t0 = time.perf_counter()
    if name == "hill":
        perm, c, *_ = hill_climb(
            instance,
            start_perm,
            neighbor_limit=params.neighbor_limit,
            max_no_improve=params.max_no_improve,
            best_improvement=True,
            rng=rng,
            progress=progress,
            time_progress=time_progress,
        )
    elif name == "tabu":
        perm, c, _ = tabu_search(
            instance,
            start_perm,
            iterations=params.tabu_iterations,
            tenure=params.tabu_tenure,
            candidate_size=params.tabu_candidate_size,
            rng=rng,
            progress=progress,
            time_progress=time_progress,
        )
    elif name == "sa":
        perm, c, *_ = simulated_annealing(
            instance,
            start_perm,
            iterations=params.sa_iterations,
            initial_temp=params.sa_initial_temp,
            cooling=params.sa_cooling,
            neighbor_moves=params.sa_neighbor_moves,
            rng=rng,
            progress=progress,
            time_progress=time_progress,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown algorithm: {name}")
    return perm, c, time.perf_counter() - t0
