"""Simulated Annealing for flow shop scheduling problem."""

import math
import random
import time
from typing import List, Tuple

from src.algorithms.base import (
    SearchState,
    get_neighbor,
    log_iteration,
    open_log_file,
)
from src.permutation_procesing import c_max


def simulated_annealing(
    processing_times: List[List[int]],
    initial_temp: float = 1000.0,
    final_temp: float = 1.0,
    alpha: float = 0.95,
    time_limit_ms: int = 100,
    neigh_mode: str = "adjacent",
    reheat_factor: float | None = None,
    stagnation_ms: int | None = None,
    temp_floor_factor: float | None = None,
    iter_log_path: str | None = None,
) -> Tuple[List[int], int, List[int], List[int]]:
    """Simulated Annealing for flow shop scheduling problem.

    Parameters:
        processing_times: m x n processing times matrix
        initial_temp: initial temperature
        final_temp: final temperature
        alpha: cooling factor (T *= alpha)
        time_limit_ms: time limit in ms
        neigh_mode: neighborhood type
        reheat_factor: reheating multiplier on stagnation (>1)
        stagnation_ms: stagnation time (ms) to trigger reheating
        temp_floor_factor: multiplier for minimum temperature (floor = final_temp * factor)
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

    T = initial_temp
    time_limit = time_limit_ms / 1000.0

    # Reheat / stagnation tracking
    last_improve_time = state.start_time
    stagnation_threshold = stagnation_ms if stagnation_ms and stagnation_ms > 0 else None
    reheat = reheat_factor if reheat_factor and reheat_factor > 1.0 else None
    floor_factor = temp_floor_factor if temp_floor_factor and temp_floor_factor >= 1.0 else 1.0
    temp_floor = final_temp * floor_factor

    with open_log_file(iter_log_path, "simulated_annealing") as log_file:
        while time.time() - state.start_time < time_limit:
            # Find best neighbor
            neighbor, neighbor_cmax, _, _ = get_neighbor(
                neigh_mode, state.current_pi, processing_times, n
            )
            delta = neighbor_cmax - state.current_cmax

            # Boltzmann acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                state.current_pi = neighbor
                state.current_cmax = neighbor_cmax

            if state.update_best():
                last_improve_time = time.time()

            log_iteration(log_file, state)
            state.iteration += 1

            # Cooling
            T = max(T * alpha, temp_floor)

            # Reheating on stagnation
            if stagnation_threshold is not None and reheat is not None:
                since_improve_ms = (time.time() - last_improve_time) * 1000.0
                if since_improve_ms >= stagnation_threshold and T < initial_temp:
                    old_T = T
                    T = min(initial_temp, T * reheat)
                    print(
                        f"[SA][reheat] stagnation {since_improve_ms:.0f}ms | "
                        f"T: {old_T:.2f} -> {T:.2f}"
                    )
                    last_improve_time = time.time()

    return state.best_pi, state.best_cmax, state.iteration_history, state.cmax_history
