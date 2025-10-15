import math
import random
import time
from typing import List

from src.dynasearch import dynasearch_full
from src.neighbors import (
    fibonahi_neighborhood,
    generate_neighbors_adjacent,
    swap_jobs,
)
from src.permutation_procesing import c_max


def tabu_search(
    processing_times: List[List[int]],
    max_time_ms: int = 100,
    tabu_tenure: int = 10,
    neigh_mode: str = "adjacent",
    iter_log_path: str | None = None,
):
    """Perform basic tabu search for flow shop scheduling problem.

    Parameters:
        processing_times: matrix m x n
        max_time_ms: czas działania algorytmu
        tabu_tenure: długość karencji dla ruchu
        neigh_mode: rodzaj sąsiedztwa
        iter_log_path: jeśli podane, na końcu każdej iteracji zapisuje w pliku CSV:
            iteration,elapsed_ms,current_cmax,best_cmax,permutation (lista liczb)
    """
    n = len(processing_times[0])
    current_pi = list(range(n))

    best_pi = current_pi.copy()
    best_cmax = c_max(best_pi, processing_times)
    current_cmax = best_cmax

    tabu_list = {}

    # Tracking convergence for plotting
    cmax_history = [best_cmax]
    iteration_history = [0]  # czas w ms od startu

    # Time tracking
    start_time = time.time()
    max_time_seconds = max_time_ms / 1000.0
    iteration = 0

    # Przygotowanie pliku logu jeśli wymagane
    log_file = None
    if iter_log_path:
        try:
            log_file = open(iter_log_path, "w", encoding="utf-8")
            log_file.write("iteration,elapsed_ms,current_cmax,best_cmax,permutation\n")
        except Exception as e:
            print(f"[tabu_search] Nie udało się otworzyć pliku logu {iter_log_path}: {e}")
            log_file = None

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

        elif neigh_mode == "dynasearch_neighborhood":
            # Zastąpiono wersję naiwną pełną implementacją dynasearch
            new_pi, new_c, _ = dynasearch_full(current_pi, processing_times)
            move_selected = tuple(new_pi)  # placeholder
            pi_selected = new_pi
            cmax_selected = new_c

        else:
            raise ValueError(f"Unknown neigh_mode={neigh_mode}")

        if pi_selected is None:
            # Brak dopuszczalnego ruchu (wszystko tabu bez aspiracji). Wymuszamy losowy ruch
            # aby zapobiec całkowitemu zablokowaniu (prosty mechanizm dywersyfikacji).
            if neigh_mode == "adjacent":
                i = random.randint(0, n - 2)
                # losowy swap sąsiadów
                neighbor = swap_jobs(current_pi, i, i + 1)
                pi_selected = neighbor
                cmax_selected = c_max(neighbor, processing_times)
                move_selected = (i, i + 1)
            else:
                # Dla innych trybów praktycznie zawsze mamy ruch, ale na wszelki wypadek
                iteration += 1
                continue

        # update current solution
        current_pi = pi_selected
        current_cmax = cmax_selected
        try:
            tabu_list[move_selected] = iteration + tabu_tenure
        except TypeError:
            # zabezpieczenie gdyby tabu_tenure było None
            tabu_list[move_selected] = iteration + 10

        # update best solution
        if cmax_selected < best_cmax:
            best_cmax = cmax_selected
            best_pi = current_pi.copy()
            cmax_history.append(best_cmax)
            elapsed_ms = int((time.time() - start_time) * 1000)
            iteration_history.append(elapsed_ms)

        # Log iteracji
        if log_file:
            try:
                elapsed_ms_full = int((time.time() - start_time) * 1000)
                permutation_str = " ".join(map(str, current_pi))
                log_file.write(
                    f"{iteration},{elapsed_ms_full},{current_cmax},"  # part1
                    f'{best_cmax},"{permutation_str}"\n'  # part2
                )
            except Exception:
                pass

        iteration += 1

    if log_file:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    return best_pi, best_cmax, iteration_history, cmax_history


def simulated_annealing(
    processing_times: List[List[int]],
    initial_temp: float = 1000.0,
    final_temp: float = 1.0,
    alpha: float = 0.95,
    time_limit_ms: int = 100,
    neigh_mode: str = "adjacent",
    # Time-based reheating parameters (reverted from iteration-based on 2025-10-05):
    reheat_factor: float | None = None,  # T *= factor (>1) on stagnation
    stagnation_ms: int | None = None,  # ms without improvement to trigger reheat
    temp_floor_factor: float | None = None,  # floor = final_temp * factor (>=1)
    iter_log_path: str | None = None,
):
    """Simulated Annealing dla problemu flow shop.

    Logowanie (jeśli podano iter_log_path) w formacie CSV:
        iteration,elapsed_ms,current_cmax,best_cmax,permutation
    Dla trybu adjacent jedna iteracja = jedno przetworzenie sąsiada (wewnątrz pętli for).
    """
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

    # Reheat / stagnation (time-based) tracking
    last_improve_time = start_time
    stagnation_ms_threshold = stagnation_ms if (stagnation_ms and stagnation_ms > 0) else None
    reheat_factor_norm = reheat_factor if (reheat_factor and reheat_factor > 1.0) else None
    temp_floor_factor_norm = (
        temp_floor_factor if (temp_floor_factor and temp_floor_factor >= 1.0) else 1.0
    )
    temp_floor = final_temp * temp_floor_factor_norm

    # Przygotowanie pliku logu jeśli wymagane
    log_file = None
    if iter_log_path:
        try:
            log_file = open(iter_log_path, "w", encoding="utf-8")
            log_file.write("iteration,elapsed_ms,current_cmax,best_cmax,permutation\n")
        except Exception as e:
            print(f"[simulated_annealing] Nie udało się otworzyć pliku logu {iter_log_path}: {e}")
            log_file = None

    while (time.time() - start_time) < time_limit:
        # Okresowe logowanie temperatury (co ~5s) bez nadmiaru spamu
        now_loop = time.time()
        if (now_loop - last_improve_time) < 0.01:  # świeża poprawa -> krótkie info
            pass
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
                    last_improve_time = time.time()

                # Log po każdej jednostkowej iteracji (po potencjalnej akceptacji / poprawie)
                if log_file:
                    try:
                        elapsed_ms_full = int((time.time() - start_time) * 1000)
                        permutation_str = " ".join(map(str, current_pi))
                        log_file.write(
                            f"{iteration},{elapsed_ms_full},{current_cmax},"
                            f'{best_cmax},"{permutation_str}"\n'
                        )
                    except Exception:
                        pass
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
                last_improve_time = time.time()

            if log_file:
                try:
                    elapsed_ms_full = int((time.time() - start_time) * 1000)
                    permutation_str = " ".join(map(str, current_pi))
                    log_file.write(
                        f"{iteration},{elapsed_ms_full},{current_cmax},"
                        f'{best_cmax},"{permutation_str}"\n'
                    )
                except Exception:
                    pass
            iteration += 1

        elif neigh_mode == "dynasearch_neighborhood":
            neighbor, neighbor_cmax, _ = dynasearch_full(current_pi, processing_times)
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
                last_improve_time = time.time()

            if log_file:
                try:
                    elapsed_ms_full = int((time.time() - start_time) * 1000)
                    permutation_str = " ".join(map(str, current_pi))
                    log_file.write(
                        f"{iteration},{elapsed_ms_full},{current_cmax},"
                        f'{best_cmax},"{permutation_str}"\n'
                    )
                except Exception:
                    pass
            iteration += 1

        else:
            raise ValueError(f"Unknown neigh_mode={neigh_mode}")

        # Cooling schedule: multiplicative
        T *= alpha
        if T < temp_floor:
            # Keep a floor temperature to continue probabilistic acceptance
            T = temp_floor

        # Time-based reheating: if no improvement for stagnation_ms_threshold
        if stagnation_ms_threshold is not None:
            since_improve_ms = (time.time() - last_improve_time) * 1000.0
            if since_improve_ms >= stagnation_ms_threshold and reheat_factor_norm is not None:
                if T < initial_temp:
                    old_T = T
                    T = min(initial_temp, T * reheat_factor_norm)
                    print(
                        f"[SA][reheat-time] stagnation {since_improve_ms:.0f}ms | "
                        f"T: {old_T:.2f} -> {T:.2f} (factor={reheat_factor_norm})"
                    )
                last_improve_time = time.time()  # reset stagnation clock after a reheat attempt

    if log_file:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    return best_pi, best_cmax, iteration_history, cmax_history
