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


def handle_tabu_move_topk(
    top_moves: List[dict],
    tabu_list: dict[Any, int],
    iteration: int,
) -> Tuple[List[int] | None, int | None, Any]:
    """Handle tabu - find first non-tabu move from top_moves.

    Works for both adjacent and fibonahi neighborhoods.

    Parameters:
        top_moves: list of dicts [{"pi": [...], "cmax": int, "move": ...}, ...] sorted ascending
        tabu_list: dictionary of tabu moves
        iteration: current iteration

    Returns:
        (new_pi, new_cmax, move_id) or (None, None, None)
    """
    if not top_moves:
        return None, None, None

    for entry in top_moves:
        move = entry["move"]
        is_tabu = move in tabu_list and tabu_list[move] > iteration
        if not is_tabu:
            return entry["pi"], entry["cmax"], move

    # All moves are tabu - take the last one (k-th, guaranteed non-tabu when k = tenure + 1)
    last = top_moves[-1]
    return last["pi"], last["cmax"], last["move"]


def handle_tabu_move(
    neigh_mode: str,
    state: SearchState,
    processing_times: List[List[int]],
    tabu_list: dict[Any, int],
    n: int,
    top_moves: List[dict] | None = None,
) -> Tuple[List[int] | None, int | None, Any]:
    """Handle situation when the best move is tabu (without aspiration).

    Behavior depends on neighborhood type:
    - adjacent: use top_moves (from get_neighbor), find first non-tabu
    - fibonahi: use top_moves (from get_neighbor), find first non-tabu
    - dynasearch: random restart (new permutation)
    - motzkin: random restart (new permutation)
    - quantum_adjacent: random restart (new permutation)
    - quantum_fibonahi: random restart (new permutation)

    Parameters:
        neigh_mode: neighborhood type
        state: current search state
        processing_times: processing times matrix
        tabu_list: dictionary of tabu moves
        n: number of jobs
        top_moves: list of top-k moves from get_neighbor (adjacent/fibonahi)

    Returns:
        (new_pi, new_cmax, move_id) or (None, None, None) if iteration should be skipped
    """
    if neigh_mode in ("adjacent", "fibonahi"):
        return handle_tabu_move_topk(top_moves, tabu_list, state.iteration)

    elif neigh_mode in ("dynasearch", "motzkin", "quantum_adjacent", "quantum_fibonahi"):
        # Restart: generate completely new random permutation
        new_pi = generate_random_permutation(n, state.current_pi)
        if new_pi is None:
            return None, None, None
        new_cmax = c_max(new_pi, processing_times)
        move_id = tuple(new_pi)  # unique restart identification
        return new_pi, new_cmax, move_id

    else:
        return None, None, None


def iterated_local_search(
    processing_times: List[List[int]],
    max_time_ms: int = 100,
    tabu_tenure: int = 10,
    neigh_mode: str = "adjacent",
    iter_log_path: str | None = None,
) -> Tuple[List[int], int, List[int], List[int]]:
    """Iterated Local Search for flow shop scheduling problem.

    Parameters:
        processing_times: m x n processing times matrix
        max_time_ms: time limit in ms
        tabu_tenure: tabu tenure (move forbidden duration)
        neigh_mode: neighborhood type
        iter_log_path: path to CSV log file

    Returns:
        (best_pi, best_cmax, iteration_history, cmax_history)
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

    with open_log_file(iter_log_path, "iterated_local_search") as log_file:
        while time.time() - state.start_time < max_time_seconds:
            # Find best neighbor (for adjacent/fibonahi: pass tenure to get top_moves)
            tabu_len = tenure if neigh_mode in ("adjacent", "fibonahi") else None
            new_pi, new_c, move_id, top_moves = get_neighbor(
                neigh_mode, state.current_pi, processing_times, n, tabu_len
            )

            # Check tabu with aspiration
            tabu_active = move_id in tabu_list and tabu_list[move_id] > state.iteration
            if tabu_active and new_c >= state.best_cmax:
                # Move is tabu and doesn't meet aspiration - handle based on neighborhood
                alt_pi, alt_c, alt_move = handle_tabu_move(
                    neigh_mode, state, processing_times, tabu_list, n, top_moves
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

            state.update_best()
            log_iteration(log_file, state)
            state.iteration += 1

    return state.best_pi, state.best_cmax, state.iteration_history, state.cmax_history
