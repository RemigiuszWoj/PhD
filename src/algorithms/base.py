"""Common structures and helper functions for search algorithms."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Tuple

from src.neighborhoods.adjacent import generate_neighbors_adjacent
from src.neighborhoods.dynasearch import dynasearch_full
from src.neighborhoods.fibonahi import fibonahi_neighborhood_topk
from src.neighborhoods.motzkin import motzkin_neighborhood_full
from src.neighborhoods.quantum_adjacent import quantum_adjacent_neighborhood
from src.neighborhoods.quantum_fibonahi import quantum_fibonahi_neighborhood
from src.permutation_procesing import c_max


@dataclass
class SearchState:
    """Shared state for search algorithms."""

    current_pi: List[int]
    current_cmax: int
    best_pi: List[int]
    best_cmax: int
    cmax_history: List[int] = field(default_factory=list)
    iteration_history: List[int] = field(default_factory=list)
    start_time: float = 0.0
    iteration: int = 0

    def update_best(self) -> bool:
        """Update best solution. Returns True if improved."""
        if self.current_cmax < self.best_cmax:
            self.best_cmax = self.current_cmax
            self.best_pi = self.current_pi.copy()
            self.cmax_history.append(self.best_cmax)
            elapsed_ms = int((time.time() - self.start_time) * 1000)
            self.iteration_history.append(elapsed_ms)
            return True
        return False

    def elapsed_ms(self) -> int:
        """Return elapsed time from start in ms."""
        return int((time.time() - self.start_time) * 1000)


@contextmanager
def open_log_file(path: str | None, algo_name: str) -> Iterator[Any]:
    """Context manager for log file handling."""
    log_file = None
    if path:
        try:
            log_file = open(path, "w", encoding="utf-8")
            log_file.write("iteration,elapsed_ms,current_cmax,best_cmax,permutation\n")
        except Exception as e:
            print(f"[{algo_name}] Failed to open log file {path}: {e}")
            log_file = None
    try:
        yield log_file
    finally:
        if log_file:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass


def log_iteration(log_file: Any, state: SearchState) -> None:
    """Write iteration to log file."""
    if log_file:
        try:
            permutation_str = " ".join(map(str, state.current_pi))
            log_file.write(
                f"{state.iteration},{state.elapsed_ms()},{state.current_cmax},"
                f'{state.best_cmax},"{permutation_str}"\n'
            )
        except Exception:
            pass


def get_neighbor(
    neigh_mode: str,
    current_pi: List[int],
    processing_times: List[List[int]],
    n: int,
    tabu_len: int | None = None,
) -> Tuple[List[int], int, Any, List[dict] | None]:
    """Generate the best neighbor for a given neighborhood mode.

    Parameters:
        neigh_mode: neighborhood type
        current_pi: current permutation
        processing_times: m x n processing times matrix
        n: number of jobs
        tabu_len: if provided, also returns top (tabu_len + 1) moves (adjacent only)

    Returns:
        (new_pi, new_cmax, move_id, top_moves)
        top_moves: list of dicts [{"pi": [...], "cmax": int, "move": (i,j)}, ...] sorted by cmax ascending
                   or None if tabu_len not provided or other neighborhood
    """
    if neigh_mode == "adjacent":
        # Collect all neighbors with their cmax
        neighbors_with_cmax = []
        for neighbor, move in generate_neighbors_adjacent(current_pi):
            c = c_max(neighbor, processing_times)
            neighbors_with_cmax.append({"pi": neighbor, "cmax": c, "move": move})

        # Sort by cmax (ascending)
        neighbors_with_cmax.sort(key=lambda x: x["cmax"])

        # Best neighbor
        best = neighbors_with_cmax[0] if neighbors_with_cmax else None
        if best is None:
            return current_pi, c_max(current_pi, processing_times), None, None

        # Top-k moves if tabu_len provided
        top_moves = None
        if tabu_len is not None:
            k = tabu_len + 1
            top_moves = neighbors_with_cmax[:k]

        return best["pi"], best["cmax"], best["move"], top_moves

    elif neigh_mode == "fibonahi":
        # Get top-k solutions (k = tabu_len + 1, or 1 if no tabu)
        k = (tabu_len + 1) if tabu_len is not None else 1
        top_moves = fibonahi_neighborhood_topk(current_pi, processing_times, k)
        if top_moves:
            best = top_moves[0]
            return best["pi"], best["cmax"], best["move"], top_moves if tabu_len else None
        return current_pi, c_max(current_pi, processing_times), None, None

    elif neigh_mode == "dynasearch":
        new_pi, new_c, _ = dynasearch_full(current_pi, processing_times)
        return new_pi, new_c, tuple(new_pi), None

    elif neigh_mode == "motzkin":
        if n > 150:
            print(f"[motzkin] Warning: n={n} may be slow; consider lower time limit.")
        new_pi, new_c, selected_pairs = motzkin_neighborhood_full(current_pi, processing_times)
        move_id = tuple(selected_pairs) if selected_pairs else tuple(new_pi)
        return new_pi, new_c, move_id, None

    elif neigh_mode == "quantum_adjacent":
        new_pi, new_c, move = quantum_adjacent_neighborhood(current_pi, processing_times)
        return new_pi, new_c, move, None

    elif neigh_mode == "quantum_fibonahi":
        new_pi, new_c, swaps = quantum_fibonahi_neighborhood(current_pi, processing_times)
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    else:
        raise ValueError(f"Unknown neigh_mode={neigh_mode}")
