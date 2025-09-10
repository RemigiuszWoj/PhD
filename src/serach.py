from typing import List

from src.neighbors import generate_neighbors_adjacent
from src.permutation_procesing import c_max


def tabu_search(processing_times: List[List[int]], max_iter: int = 100, tabu_tenure: int = 10):
    """Perform basic tabu search for flow shop scheduling problem."""
    n = len(processing_times[0])
    current_pi = list(range(n))

    best_pi = current_pi.copy()
    best_cmax = c_max(best_pi, processing_times)

    tabu_list = {}

    for iteration in range(max_iter):
        neighbors = generate_neighbors_adjacent(current_pi)
        move_selected = None
        pi_selected = None
        cmax_selected = float("inf")

        # choose best admissible neighbor
        for neighbor, move in neighbors:
            c = c_max(neighbor, processing_times)
            tabu_active = move in tabu_list and tabu_list[move] > iteration

            if tabu_active and c >= best_cmax:
                continue  # move forbidden

            if c < cmax_selected:
                cmax_selected = c
                pi_selected = neighbor
                move_selected = move

        if pi_selected is None:
            break  # no admissible move found

        # update current solution
        current_pi = pi_selected
        tabu_list[move_selected] = iteration + tabu_tenure

        # update best solution
        if cmax_selected < best_cmax:
            best_cmax = cmax_selected
            best_pi = current_pi.copy()

    return best_pi, best_cmax
