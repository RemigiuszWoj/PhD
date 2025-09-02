import random
import time

from src.decoder import build_schedule_from_permutation
from src.models import DataInstance, OperationKey, Schedule
from src.neighborhood import create_fibonachi_neighborhood, swap_adjacent


def evaluate(
    data: DataInstance,
    permutation: list[OperationKey],
    cache: dict | None = None,
    return_schedule: bool = False,
    validate: bool = False,
) -> int | tuple[int, Schedule]:
    key = tuple(permutation)
    if cache is not None and key in cache:
        cmax, sched = cache[key]
        if return_schedule:
            return cmax, sched
        return cmax
    sched = build_schedule_from_permutation(
        data,
        key,
        validate=validate,
        check_completeness=True,
    )
    if cache is not None:
        cache[key] = (sched.cmax, sched)
    if return_schedule:
        return sched.cmax, sched
    return sched.cmax


def tabu_search(
    data: DataInstance,
    start_perm: list[OperationKey],
    iterations: int = 200,
    tenure: int = 15,
    candidate_size: int = 60,
    rng: random.Random | None = None,
    cache: dict | None = None,
    aspiration: bool = True,
    progress: list[int] | None = None,
    time_progress: list[float] | None = None,
    neighborhood: str = "swap_adjacent_neighborhood",
) -> tuple[list[OperationKey], int, int]:
    """Tabu Search z jedynym sąsiedztwem: adjacent swap (zamiana i,i+1 różnych jobów).

    candidate_size: maksymalna liczba indeksów i (dla par (i,i+1)) rozważanych w iteracji.
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
    if progress is not None:
        progress.append(best_c)
    if time_progress is not None:
        time_progress.append(0.0)
    for it in range(1, iterations + 1):
        n = len(current)
        best_move_key = None
        best_move_perm = None
        best_move_c = None

        # Neighborhood generation block
        if neighborhood == "swap_adjacent_neighborhood":  # adjacent swap neighborhood
            indices = list(range(n - 1))
            if candidate_size < len(indices):
                indices = rng.sample(indices, candidate_size)
            for i in indices:
                new_perm = swap_adjacent(current, i)
                if new_perm == current:
                    continue
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
            # placeholder for future neighborhood strategy names
            pass

        if best_move_perm is None:
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
        if progress is not None:
            progress.append(best_c)
        if time_progress is not None:
            time_progress.append(time.perf_counter() - t0)

    return best_perm, best_c, evals


def simulated_annealing(
    data: DataInstance,
    start_perm: list[OperationKey],
    iterations: int = 1000,
    initial_temp: float = 50.0,
    cooling: float = 0.95,
    neighbor_moves: int = 1,
    rng: random.Random | None = None,
    cache: dict | None = None,
    min_temp: float = 1e-3,
    progress: list[int] | None = None,
    time_progress: list[float] | None = None,
    neighborhood: str = "swap_adjacent_neighborhood",
    # neighborhood: str = "fibonachi_neighborhood",
) -> tuple[list[OperationKey], int, int, int]:
    import math

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
    if progress is not None:
        progress.append(best_c)
    if time_progress is not None:
        time_progress.append(0.0)
    it = 0
    while it < iterations and T > min_temp:
        it += 1
        candidate_best_perm: list[OperationKey] | None = None
        candidate_best_c: int | None = None

        for _ in range(neighbor_moves):
            if len(current) < 2:
                neigh = current[:]
            else:
                if neighborhood == "swap_adjacent_neighborhood":  # adjacent swap neighborhood
                    neigh = current[:]
                    for _try in range(10):
                        idx = rng.randrange(0, len(current) - 1)
                        cand = swap_adjacent(current, idx)
                        if cand != current:
                            neigh = cand
                            break
                elif neighborhood == "fibonachi_neighborhood":
                    # placeholder: pojedyncza transformacja permutacji
                    neigh = create_fibonachi_neighborhood(current[:])
                    print(f"Fibonachi neighborhood generated: {neigh}")
                    raise NotImplementedError("Fibonachi neighborhood not implemented yet")

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
        if delta < 0:
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
        T *= cooling
        if progress is not None:
            progress.append(best_c)
        if time_progress is not None:
            time_progress.append(time.perf_counter() - t0)
    return best_perm, best_c, evals, it
