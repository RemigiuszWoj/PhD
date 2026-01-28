"""Moduł algorytmów przeszukiwania: Tabu Search i Simulated Annealing.

Zrefaktoryzowana wersja z wydzielonymi funkcjami pomocniczymi.
"""

import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Tuple

from src.dynasearch import dynasearch_full
from src.motzkin import motzkin_neighborhood_full
from src.neighbors import (
    fibonahi_neighborhood,
    generate_neighbors_adjacent,
    swap_jobs,
)
from src.permutation_procesing import c_max
from src.quantum_neighbors import generate_neighbors_adjacent_qubo


# ---------------------------------------------------------------------------
# Pomocnicze struktury i funkcje
# ---------------------------------------------------------------------------


@dataclass
class SearchState:
    """Stan wspólny dla algorytmów przeszukiwania."""

    current_pi: List[int]
    current_cmax: int
    best_pi: List[int]
    best_cmax: int
    cmax_history: List[int] = field(default_factory=list)
    iteration_history: List[int] = field(default_factory=list)
    start_time: float = 0.0
    iteration: int = 0

    def update_best(self) -> bool:
        """Aktualizuje najlepsze rozwiązanie jeśli obecne jest lepsze. Zwraca True jeśli poprawiono."""
        if self.current_cmax < self.best_cmax:
            self.best_cmax = self.current_cmax
            self.best_pi = self.current_pi.copy()
            self.cmax_history.append(self.best_cmax)
            elapsed_ms = int((time.time() - self.start_time) * 1000)
            self.iteration_history.append(elapsed_ms)
            return True
        return False

    def elapsed_ms(self) -> int:
        """Zwraca czas od startu w ms."""
        return int((time.time() - self.start_time) * 1000)


@contextmanager
def open_log_file(path: str | None, algo_name: str) -> Iterator[Any]:
    """Context manager do obsługi pliku logu."""
    log_file = None
    if path:
        try:
            log_file = open(path, "w", encoding="utf-8")
            log_file.write("iteration,elapsed_ms,current_cmax,best_cmax,permutation\n")
        except Exception as e:
            print(f"[{algo_name}] Nie udało się otworzyć pliku logu {path}: {e}")
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
    """Zapisuje iterację do pliku logu."""
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
) -> Tuple[List[int], int, Any]:
    """Generuje sąsiada dla danego trybu.

    Zwraca: (new_pi, new_cmax, move_id)
    """
    if neigh_mode == "fibonahi_neighborhood":
        new_pi, new_c = fibonahi_neighborhood(current_pi, processing_times)
        return new_pi, new_c, tuple(new_pi)

    elif neigh_mode == "dynasearch_neighborhood":
        new_pi, new_c, _ = dynasearch_full(current_pi, processing_times)
        return new_pi, new_c, tuple(new_pi)

    elif neigh_mode == "motzkin_neighborhood":
        if n > 150:
            print(f"[motzkin] Warning: n={n} may be slow; consider lower time limit.")
        new_pi, new_c, selected_pairs = motzkin_neighborhood_full(current_pi, processing_times)
        move_id = tuple(selected_pairs) if selected_pairs else tuple(new_pi)
        return new_pi, new_c, move_id

    elif neigh_mode == "quantum_adjacent":
        new_pi, move = generate_neighbors_adjacent_qubo(current_pi, processing_times)
        new_c = c_max(new_pi, processing_times)
        return new_pi, new_c, move

    else:
        raise ValueError(f"Unknown neigh_mode={neigh_mode}")


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------


def tabu_search(
    processing_times: List[List[int]],
    max_time_ms: int = 100,
    tabu_tenure: int = 10,
    neigh_mode: str = "adjacent",
    iter_log_path: str | None = None,
) -> Tuple[List[int], int, List[int], List[int]]:
    """Tabu Search dla problemu flow shop.

    Parameters:
        processing_times: macierz m x n czasów przetwarzania
        max_time_ms: limit czasu w ms
        tabu_tenure: długość karencji dla ruchu
        neigh_mode: rodzaj sąsiedztwa
        iter_log_path: ścieżka do pliku logu CSV

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

    with open_log_file(iter_log_path, "tabu_search") as log_file:
        while time.time() - state.start_time < max_time_seconds:
            move_selected = None
            pi_selected = None
            cmax_selected = float("inf")

            if neigh_mode == "adjacent":
                # Wybór najlepszego dopuszczalnego sąsiada
                for neighbor, move in generate_neighbors_adjacent(state.current_pi):
                    c = c_max(neighbor, processing_times)
                    tabu_active = move in tabu_list and tabu_list[move] > state.iteration

                    if tabu_active and c >= state.best_cmax:
                        continue  # ruch zabroniony (brak aspiracji)

                    if c < cmax_selected:
                        cmax_selected = c
                        pi_selected = neighbor
                        move_selected = move
            else:
                # Pozostałe sąsiedztwa: jeden ruch na iterację
                new_pi, new_c, move_id = get_neighbor(
                    neigh_mode, state.current_pi, processing_times, n
                )
                tabu_active = move_id in tabu_list and tabu_list[move_id] > state.iteration

                if tabu_active and new_c >= state.best_cmax:
                    # Ruch tabu bez aspiracji - pomiń iterację
                    state.iteration += 1
                    continue

                pi_selected = new_pi
                cmax_selected = new_c
                move_selected = move_id

            # Fallback: brak dopuszczalnego ruchu
            if pi_selected is None:
                if neigh_mode == "adjacent":
                    i = random.randint(0, n - 2)
                    neighbor = swap_jobs(state.current_pi, i, i + 1)
                    pi_selected = neighbor
                    cmax_selected = c_max(neighbor, processing_times)
                    move_selected = (i, i + 1)
                else:
                    state.iteration += 1
                    continue

            # Aktualizacja stanu
            state.current_pi = pi_selected
            state.current_cmax = cmax_selected
            tabu_list[move_selected] = state.iteration + tenure

            state.update_best()
            log_iteration(log_file, state)
            state.iteration += 1

    return state.best_pi, state.best_cmax, state.iteration_history, state.cmax_history


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------


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
    """Simulated Annealing dla problemu flow shop.

    Parameters:
        processing_times: macierz m x n czasów przetwarzania
        initial_temp: temperatura początkowa
        final_temp: temperatura końcowa
        alpha: współczynnik chłodzenia (T *= alpha)
        time_limit_ms: limit czasu w ms
        neigh_mode: rodzaj sąsiedztwa
        reheat_factor: mnożnik podgrzewania przy stagnacji (>1)
        stagnation_ms: czas stagnacji (ms) do wywołania podgrzewania
        temp_floor_factor: mnożnik dla minimalnej temperatury (floor = final_temp * factor)
        iter_log_path: ścieżka do pliku logu CSV

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
            if neigh_mode == "adjacent":
                # Dla adjacent: n losowych prób na jedną temperaturę
                for _ in range(n):
                    i = random.randint(0, n - 2)
                    neighbor = swap_jobs(state.current_pi, i, i + 1)
                    neighbor_cmax = c_max(neighbor, processing_times)
                    delta = neighbor_cmax - state.current_cmax

                    if delta < 0 or random.random() < math.exp(-delta / T):
                        state.current_pi = neighbor
                        state.current_cmax = neighbor_cmax

                    if state.update_best():
                        last_improve_time = time.time()

                    log_iteration(log_file, state)
                    state.iteration += 1
            else:
                # Pozostałe sąsiedztwa: jeden ruch na iterację
                neighbor, neighbor_cmax, _ = get_neighbor(
                    neigh_mode, state.current_pi, processing_times, n
                )
                delta = neighbor_cmax - state.current_cmax

                if delta < 0 or random.random() < math.exp(-delta / T):
                    state.current_pi = neighbor
                    state.current_cmax = neighbor_cmax

                if state.update_best():
                    last_improve_time = time.time()

                log_iteration(log_file, state)
                state.iteration += 1

            # Chłodzenie
            T = max(T * alpha, temp_floor)

            # Podgrzewanie przy stagnacji
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
