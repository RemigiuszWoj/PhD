import math
import random
import time
from typing import List

from src.neighbors import fibonahi_neighborhood, generate_neighbors_adjacent, swap_jobs
from src.permutation_procesing import c_max


def tabu_search(
    processing_times: List[List[int]],
    max_time_ms: int = 100,
    tabu_tenure: int = 10,
    neigh_mode: str = "adjacent",
):
    """Perform basic tabu search for flow shop scheduling problem."""
    n = len(processing_times[0])
    current_pi = list(range(n))

    best_pi = current_pi.copy()
    best_cmax = c_max(best_pi, processing_times)

    tabu_list = {}

    # Tracking convergence for plotting
    cmax_history = [best_cmax]
    iteration_history = [0]  # czas w ms od startu

    # Time tracking
    start_time = time.time()
    max_time_seconds = max_time_ms / 1000.0
    iteration = 0

    while time.time() - start_time < max_time_seconds:
        move_selected = None
        pi_selected = None
        cmax_selected = float("inf")
        if neigh_mode == "adjacent":
            # choose best admissible neighbor
            neighbors = generate_neighbors_adjacent(current_pi)
            for neighbor, move in neighbors:
                c = c_max(neighbor, processing_times)
                tabu_active = move in tabu_list and tabu_list[move] > iteration

                if tabu_active and c >= best_cmax:
                    continue  # move forbidden

                if c < cmax_selected:
                    cmax_selected = c
                    pi_selected = neighbor
                    move_selected = move

        elif neigh_mode == "fibonahi_neighborhood":
            new_pi, new_c = fibonahi_neighborhood(current_pi, processing_times)
            move_selected = tuple(new_pi)  # just a placeholder to store in tabu
            pi_selected = new_pi
            cmax_selected = new_c

        else:
            raise ValueError(f"Unknown neigh_mode={neigh_mode}")

        if pi_selected is None:
            break  # no admissible move found

        # update current solution
        current_pi = pi_selected
        tabu_list[move_selected] = iteration + tabu_tenure

        # update best solution
        if cmax_selected < best_cmax:
            best_cmax = cmax_selected
            best_pi = current_pi.copy()
            cmax_history.append(best_cmax)
            elapsed_ms = int((time.time() - start_time) * 1000)
            iteration_history.append(elapsed_ms)

        iteration += 1

    return best_pi, best_cmax, iteration_history, cmax_history


def simulated_annealing(
    processing_times: List[List[int]],
    initial_temp: float = 1000.0,
    final_temp: float = 1.0,
    alpha: float = 0.95,
    time_limit_ms: int = 100,
    neigh_mode: str = "adjacent",
):
    """Simulated Annealing for Flow Shop starting from initial_pi."""
    n = len(processing_times[0])
    current_pi = list(range(n))
    current_cmax = c_max(current_pi, processing_times)

    best_pi = current_pi.copy()
    best_cmax = current_cmax

    # Tracking convergence for plotting
    cmax_history = [best_cmax]
    iteration_history = [0]  # czas w ms od startu

    T = initial_temp
    start_time = time.time()
    time_limit = time_limit_ms / 1000
    iteration = 0

    while T > final_temp and (time.time() - start_time) < time_limit:
        if neigh_mode == "adjacent":
            for _ in range(n):
                i = random.randint(0, n - 2)
                neighbor = swap_jobs(current_pi, i, i + 1)
                neighbor_cmax = c_max(neighbor, processing_times)
                delta = neighbor_cmax - current_cmax

                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_pi = neighbor
                    current_cmax = neighbor_cmax

                if current_cmax < best_cmax:
                    best_cmax = current_cmax
                    best_pi = current_pi.copy()
                    cmax_history.append(best_cmax)
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    iteration_history.append(elapsed_ms)

                iteration += 1

        elif neigh_mode == "fibonahi_neighborhood":
            neighbor, neighbor_cmax = fibonahi_neighborhood(current_pi, processing_times)
            delta = neighbor_cmax - current_cmax

            if delta < 0 or random.random() < math.exp(-delta / T):
                current_pi = neighbor
                current_cmax = neighbor_cmax

            if current_cmax < best_cmax:
                best_cmax = current_cmax
                best_pi = current_pi.copy()
                cmax_history.append(best_cmax)
                elapsed_ms = int((time.time() - start_time) * 1000)
                iteration_history.append(elapsed_ms)

            iteration += 1

        else:
            raise ValueError(f"Unknown neigh_mode={neigh_mode}")

        T *= alpha

    return best_pi, best_cmax, iteration_history, cmax_history
