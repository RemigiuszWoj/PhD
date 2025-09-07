import logging
import math
import os
import random
import time
from typing import Optional

from src.evaluation import evaluate
from src.models import DataInstance, OperationKey
from src.neighborhood import create_fibonachi_neighborhood, swap_adjacent
from src.visualization import plot_gantt


def tabu_search(
    data: DataInstance,
    start_perm: list[OperationKey],
    iterations: int = 200,
    tenure: int = 15,
    rng: Optional[random.Random] = None,
    cache: Optional[dict] = None,
    aspiration: bool = True,
    neighborhood: str = "swap_adjacent_neighborhood",
    time_limit_ms: Optional[int] = None,
    gantt_dir: Optional[str] = None,
    trace_file: str | None = None,
) -> tuple[list[OperationKey], int, int, list[int]]:
    """Tabu Search z jedynym sąsiedztwem: adjacent swap (zamiana i,i+1 różnych jobów).

    (upr.) pełne sąsiedztwo adjacent bez limitu liczby kandydatów.
    time_limit_ms: opcjonalny limit czasu działania (ms);
        jeśli ustawiony, przerywamy po przekroczeniu.
    save_gantt_dir: katalog do zapisu wykresów Gantta z każdej iteracji (lub None).
    save_progress_path: ścieżka do zapisu końcowego wykresu postępu (lub None).
    progress_plot: czy generować końcowy wykres postępu (wymaga progress oraz ścieżki).
    """
    # Initialize random number generator if not provided
    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}
    current = list(start_perm)
    current_c = evaluate(data, current, cache=cache)
    best_perm = list(current)
    best_c = current_c
    evals = 1

    # tabu klucz: posortowana para operacji (stabilnie odzwierciedla ruch)
    tabu: dict[tuple[OperationKey, OperationKey], int] = {}

    t0 = time.perf_counter()
    current_history: list[int] = []  # surowy current c po każdej zmianie
    # Dodajemy wartość początkową aby wykres porównawczy nie był pusty gdy brak ruchów
    current_history.append(current_c)
    # progress list usunięty – używamy tylko current_history
    limit_s = (time_limit_ms / 1000.0) if time_limit_ms is not None else None
    # ensure directories
    if gantt_dir is not None:
        os.makedirs(gantt_dir, exist_ok=True)
    for it in range(1, iterations + 1):
        if limit_s is not None and (time.perf_counter() - t0) >= limit_s:
            logging.getLogger("jssp").info(
                "[tabu] stop time_limit reached at iter %d best=%s evals=%d", it, best_c, evals
            )
            break
        n = len(current)
        best_move_key = None
        best_move_perm = None
        best_move_c = None

        # Neighborhood generation block
        if neighborhood == "swap_adjacent_neighborhood":  # adjacent swap neighborhood
            # Bierzemy wyłącznie pozycje gdzie joby się różnią (realne ruchy)
            indices = [i for i in range(n - 1) if current[i][0] != current[i + 1][0]]
            if not indices:  # brak możliwych ruchów
                logging.getLogger("jssp").info(
                    "[tabu] brak ruchów w iter %d best=%s evals=%d", it, best_c, evals
                )
                break
            for i in indices:
                # tu swap zawsze zmieni permutację (bo joby różne)
                new_perm = swap_adjacent(current, i)
                op_a, op_b = sorted((current[i], current[i + 1]))
                move_key = (op_a, op_b)
                c = evaluate(data, new_perm, cache=cache)
                evals += 1
                is_tabu = move_key in tabu and tabu[move_key] >= it
                if is_tabu and not (aspiration and c < best_c):
                    continue
                if best_move_c is None or c < best_move_c:
                    best_move_c = c
                    best_move_key = move_key
                    best_move_perm = new_perm

        elif neighborhood == "fibonachi_neighborhood":
            new_perm, c = create_fibonachi_neighborhood(
                current,
                data=data,
                cache=cache,
                allow_equal=True,
                debug=False,
                return_cost=True,
            )
            evals += 1
            if new_perm != current:
                swapped_pairs: list[tuple[OperationKey, OperationKey]] = []
                for i in range(n - 1):
                    if (
                        new_perm[i] == current[i + 1]
                        and new_perm[i + 1] == current[i]
                        and current[i][0] != current[i + 1][0]
                    ):
                        op_a, op_b = sorted((current[i], current[i + 1]))
                        swapped_pairs.append((op_a, op_b))
                if swapped_pairs:
                    if len(swapped_pairs) == 1:
                        move_key: tuple | tuple[OperationKey, OperationKey] = swapped_pairs[0]
                    else:
                        move_key = tuple(swapped_pairs)
                    is_tabu = move_key in tabu and tabu[move_key] >= it
                    if not (is_tabu and not (aspiration and c < best_c)):
                        if best_move_c is None or c < best_move_c:
                            best_move_c = c
                            best_move_key = move_key
                            best_move_perm = new_perm

        if best_move_perm is None:
            logging.getLogger("jssp").info(
                "[tabu] brak akceptowalnego ruchu iter=%d best=%s evals=%d", it, best_c, evals
            )
            break
        current = best_move_perm
        current_c = best_move_c if best_move_c is not None else current_c
        if best_move_key is not None:
            tabu[best_move_key] = it + tenure
        expired = [k for k, exp in tabu.items() if exp < it]
        for k in expired:
            del tabu[k]
        if current_c < best_c:
            best_c = int(current_c)
            best_perm = list(current)
        current_history.append(current_c)
        logging.getLogger("jssp").info(
            "[tabu] iter %d/%d current=%s best=%s evals=%d",
            it,
            iterations,
            current_c,
            best_c,
            evals,
        )
        if trace_file is not None:
            # format: it;current;best;move_key;evals
            try:
                with open(trace_file, "a", encoding="utf-8") as tf:
                    tf.write(f"{it};{current_c};{best_c};{best_move_key};{evals}\n")
            except Exception:
                pass
        # zapis Gantta (zawsze jeśli katalog podany)
        if gantt_dir is not None:
            c_sched = evaluate(data, current, cache=cache, return_schedule=True)
            if isinstance(c_sched, tuple):
                _, sched = c_sched
                plot_gantt(
                    sched,
                    save_path=f"{gantt_dir}/tabu_iter_{it}.png",
                    algo_name="tabu",
                    white_background=True,
                    add_legend=False,
                )

    return best_perm, best_c, evals, current_history


def simulated_annealing(
    data: DataInstance,
    start_perm: list[OperationKey],
    iterations: int = 1000,
    initial_temp: float = 50.0,
    cooling: float = 0.95,
    neighbor_moves: int = 1,
    rng: Optional[random.Random] = None,
    cache: Optional[dict] = None,
    min_temp: float = 1e-3,
    neighborhood: str = "swap_adjacent_neighborhood",
    time_limit_ms: Optional[int] = None,
    gantt_dir: Optional[str] = None,
    trace_file: str | None = None,
) -> tuple[list[OperationKey], int, int, int, list[int]]:

    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}
    current = list(start_perm)
    current_c = evaluate(data, current, cache=cache)
    best_perm = list(current)
    best_c = int(current_c)
    T = float(initial_temp)
    evals = 1

    t0 = time.perf_counter()
    current_history: list[int] = []
    # brak osobnej listy best_c – wystarczy current_history dla przebiegu
    it = 0
    limit_s = (time_limit_ms / 1000.0) if time_limit_ms is not None else None
    if gantt_dir is not None:
        os.makedirs(gantt_dir, exist_ok=True)
    while it < iterations and T > min_temp:
        if limit_s is not None and (time.perf_counter() - t0) >= limit_s:
            logging.getLogger("jssp").info(
                "[sa] stop time_limit reached at iter %d best=%s evals=%d", it, best_c, evals
            )
            break
        it += 1
        candidate_best_perm: list[OperationKey] | None = None
        candidate_best_c: int | None = None
        move_repr: str | None = None

        if neighborhood == "fibonachi_neighborhood":
            # deterministyczny wieloswap – jeden ruch wystarczy
            if len(current) < 2:
                candidate_best_perm = current[:]
                candidate_best_c = current_c
            else:
                neigh, c = create_fibonachi_neighborhood(
                    current[:],
                    data=data,
                    cache=cache,
                    allow_equal=True,
                    debug=False,
                    return_cost=True,
                )
                candidate_best_perm = neigh
                candidate_best_c = c
                evals += 1
        else:
            # standardowe wielokrotne próbkowanie pojedynczych swapów
            for _ in range(neighbor_moves):
                if len(current) < 2:
                    neigh = current[:]
                    c = current_c
                else:
                    neigh = current[:]
                    for _try in range(10):
                        idx = rng.randrange(0, len(current) - 1)
                        cand = swap_adjacent(current, idx)
                        if cand != current:
                            neigh = cand
                            break
                    c = evaluate(data, neigh, cache=cache)
                evals += 1
                if candidate_best_c is None or c < candidate_best_c:
                    candidate_best_c = c
                    candidate_best_perm = neigh
        if candidate_best_perm is None or candidate_best_c is None:
            T *= cooling
            continue
        delta = candidate_best_c - current_c
        accept = False
        if delta <= 0:
            accept = True
        elif T > 0:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                accept = True
        if accept and candidate_best_perm is not None and candidate_best_c is not None:
            current = candidate_best_perm
            current_c = candidate_best_c
            if current_c < best_c:
                best_c = int(current_c)
                best_perm = list(current)
        prev_T = T
        T *= cooling
        current_history.append(current_c)
        logging.getLogger("jssp").info(
            "[sa] iter %d/%d T=%.4f->%.4f current=%s best=%s delta=%s acc=%d evals=%d",
            it,
            iterations,
            prev_T,
            T,
            current_c,
            best_c,
            delta,
            1 if accept else 0,
            evals,
        )
        if trace_file is not None:
            # it;T;current;best;delta;accepted;evals;move
            try:
                with open(trace_file, "a", encoding="utf-8") as tf:
                    tf.write(
                        f"{it};{T:.6f};{current_c};{best_c};{delta};{int(accept)};{evals};"
                        f"{move_repr}\n"
                    )
            except Exception:
                pass
        if gantt_dir is not None:
            c_sched = evaluate(data, current, cache=cache, return_schedule=True)
            if isinstance(c_sched, tuple):
                _, sched = c_sched
                plot_gantt(
                    sched,
                    save_path=f"{gantt_dir}/sa_iter_{it}.png",
                    algo_name="sa",
                    white_background=True,
                    add_legend=False,
                )
    return best_perm, best_c, evals, it, current_history
