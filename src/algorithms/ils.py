"""Iterated Local Search for flow shop scheduling problem."""

import random
import time
from typing import Any, List, Tuple

from src.algorithms.base import (
    SearchState,
    get_neighbor,
    log_iteration,
    open_log_file,
)
from src.algorithms.mushroom_list import MushroomList
from src.permutation_procesing import c_max


def generate_random_permutation(
    n: int, current_pi: List[int], max_attempts: int = 100
) -> List[int] | None:
    """Generate a random permutation different from the current one.

    Parameters:
        n: number of jobs
        current_pi: current permutation (to avoid)
        max_attempts: maximum number of attempts

    Returns:
        New permutation or None if failed to generate a different one
    """
    for _ in range(max_attempts):
        new_pi = list(range(n))
        random.shuffle(new_pi)
        if new_pi != current_pi:
            return new_pi
    return None


def handle_tabu_move(
    state: SearchState,
    processing_times: List[List[int]],
    n: int,
    mushroom_list: MushroomList | None = None,
) -> Tuple[List[int] | None, int | None, Any]:
    """Diversify when the best move is tabu and aspiration is not met.

    Applied uniformly to every neighborhood: perturb an elite solution
    from the MushroomList with a double-bridge kick, falling back to a
    random restart when the elite pool is still empty. This is the
    "Iterated" part of Iterated Local Search — without it the search
    cannot escape a local optimum once every productive move is tabu.

    Parameters:
        state: current search state
        processing_times: processing times matrix
        n: number of jobs
        mushroom_list: elite pool for diversification

    Returns:
        (new_pi, new_cmax, move_id) or (None, None, None) if no
        diversified solution could be produced.
    """
    new_pi = None
    if mushroom_list is not None and len(mushroom_list) > 0:
        new_pi = mushroom_list.perturb()
    if new_pi is None:
        new_pi = generate_random_permutation(n, state.current_pi)
    if new_pi is None:
        return None, None, None
    new_cmax = c_max(new_pi, processing_times)
    move_id = tuple(new_pi)
    return new_pi, new_cmax, move_id


def iterated_local_search(
    processing_times: List[List[int]],
    max_time_ms: int = 100,
    tabu_tenure: int = 10,
    neigh_mode: str = "adjacent",
    iter_log_path: str | None = None,
    quantum_config: dict | None = None,
    mushroom_k: int = 10,
) -> Tuple[List[int], int, List[int], List[int], dict]:
    """Iterated Local Search for flow shop scheduling problem.

    Parameters:
        processing_times: m x n processing times matrix
        max_time_ms: time limit in ms
        tabu_tenure: tabu tenure (move forbidden duration)
        neigh_mode: neighborhood type
        iter_log_path: path to CSV log file
        quantum_config: optional dict with quantum params (num_reads, L_max_dynasearch, etc.)
        mushroom_k: elite pool size for diversification (MushroomList)

    Returns:
        (best_pi, best_cmax, iteration_history, cmax_history, stats)
        where stats = {"iterations": int, "neigh_time_ms": int} —
        loop passes and cumulative wall time inside get_neighbor().
    """
    n = len(processing_times[0])
    initial_pi = list(range(n))
    initial_cmax = c_max(initial_pi, processing_times)

    state = SearchState(
        current_pi=initial_pi,
        current_cmax=initial_cmax,
        best_pi=initial_pi.copy(),
        best_cmax=initial_cmax,
        cmax_history=[initial_cmax],
        iteration_history=[0],
        start_time=time.time(),
        iteration=0,
    )

    tabu_list: dict[Any, int] = {}
    max_time_seconds = max_time_ms / 1000.0
    tenure = tabu_tenure if tabu_tenure else 10
    mushroom_list = MushroomList(k=mushroom_k)
    mushroom_list.offer(initial_pi, initial_cmax)

    neigh_time_s = 0.0  # cumulative wall time inside get_neighbor()

    with open_log_file(iter_log_path, "iterated_local_search") as log_file:
        while time.time() - state.start_time < max_time_seconds:
            # Find the best neighbor of the current solution.
            _t0 = time.time()
            new_pi, new_c, move_id, _ = get_neighbor(
                neigh_mode, state.current_pi, processing_times, n, None, quantum_config
            )
            neigh_time_s += time.time() - _t0

            # Check tabu with aspiration
            tabu_active = move_id in tabu_list and tabu_list[move_id] > state.iteration
            if tabu_active and new_c >= state.best_cmax:
                # Move is tabu and aspiration not met — diversify (ILS kick).
                alt_pi, alt_c, alt_move = handle_tabu_move(
                    state, processing_times, n, mushroom_list
                )
                if alt_pi is None:
                    # No alternative - skip iteration
                    state.iteration += 1
                    continue
                # Use alternative move
                new_pi, new_c, move_id = alt_pi, alt_c, alt_move

            # Update state
            state.current_pi = new_pi
            state.current_cmax = new_c
            tabu_list[move_id] = state.iteration + tenure

            prev_best = state.best_cmax
            state.update_best()
            if state.best_cmax < prev_best:
                mushroom_list.offer(state.best_pi, state.best_cmax)

            log_iteration(log_file, state)
            state.iteration += 1

    stats = {
        "iterations": state.iteration,
        "neigh_time_ms": int(neigh_time_s * 1000),
    }
    return state.best_pi, state.best_cmax, state.iteration_history, state.cmax_history, stats
