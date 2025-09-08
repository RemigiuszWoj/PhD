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
    enable_diversification: bool = True,
) -> tuple[list[OperationKey], int, int, list[int]]:
    """Tabu Search core loop.

    Notes:
        - Neighborhood currently one of: 'swap_adjacent_neighborhood', 'fibonachi_neighborhood'.
        - Full best-improving scan of the chosen neighborhood (no candidate cap).
        - Aspiration: tabu move accepted if it improves global best.
        - Frequency-based long-term memory penalises repeated moves after stagnation threshold.
        - Diversification: after prolonged stagnation perform random perturbations and reset memory.
        - Fallback: if all moves tabu, pick least bad tabu (adjusted cost then earliest expiry).

    Args (selected):
        iterations: Max main iterations.
        tenure: Base tabu tenure (dynamically lifted after diversification, reset on improvement).
        time_limit_ms: Optional wall-clock limit (milliseconds).
        gantt_dir: If provided, per-iteration Gantt charts (costly for large instances).
        trace_file: Optional CSV-like trace file (appended each iteration).
        enable_diversification: If False, disables the stagnation-triggered diversification block.
    """
    # RNG restored (diversification / random fallback decisions).
    if rng is None:
        rng = random.Random()
    if cache is None:
        cache = {}
    current = list(start_perm)
    current_c = evaluate(data, current, cache=cache)
    best_perm = list(current)
    best_c = current_c
    evals = 1

    # Tabu list: maps move key (single pair or tuple of pairs for multi-swap) -> expiry iteration
    tabu: dict[tuple, int] = {}
    # Frequency (long-term) memory for penalising over-used moves
    freq: dict[tuple[OperationKey, OperationKey], int] = {}
    # Diversification / stagnation parameters
    base_tenure = tenure
    current_tenure = tenure
    last_improvement_iter = 0
    # iterations without improvement triggering diversification
    STAGN_LIMIT = max(25, iterations // 8)
    DIVERSIFICATION_SWAPS = 6  # number of random adjacent perturbations during diversification
    # Activate frequency penalty past mid stagnation horizon
    FREQ_PENALTY_ACTIVE_AFTER = STAGN_LIMIT // 2
    freq_weight = 0.0  # dynamicznie zmieniane

    t0 = time.perf_counter()
    current_history: list[int] = []  # raw current cost after each accepted move
    # Seed history with initial value for plotting uniform length across runs
    current_history.append(current_c)
    # Removed separate 'progress' list – current history sufficient for plots
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
        used_tabu_fallback = False

        # Diversification trigger if stagnation >= threshold
        stagn_iters = it - last_improvement_iter
        if enable_diversification and stagn_iters >= STAGN_LIMIT:
            logging.getLogger("jssp").info(
                "[tabu] diversification trigger iter=%d stagnation=%d (tenure=%d)",
                it,
                stagn_iters,
                current_tenure,
            )
            # Several random real adjacent swaps (if any feasible)
            for _ in range(DIVERSIFICATION_SWAPS):
                adj_indices = [i for i in range(n - 1) if current[i][0] != current[i + 1][0]]
                if not adj_indices:
                    break
                idx = rng.choice(adj_indices)
                new_cand = swap_adjacent(current, idx)
                if new_cand != current:
                    current = new_cand
                    current_c = evaluate(data, current, cache=cache)
                    evals += 1
            # Clear tabu list (simple full reset)
            tabu.clear()
            # Lift tenure slightly to delay revisiting old regions
            current_tenure = min(base_tenure + 3, base_tenure + 6)
            last_improvement_iter = it  # reset stagnation counter
            current_history.append(current_c)
            if trace_file is not None:
                try:
                    with open(trace_file, "a", encoding="utf-8") as tf:
                        tf.write(f"{it};{current_c};{best_c};DIVERSIFY;{evals}\n")
                except Exception:
                    pass
            continue  # proceed to next outer iteration post diversification

        # Activate frequency penalty if stagnation grows
        if stagn_iters >= FREQ_PENALTY_ACTIVE_AFTER:
            freq_weight = 1.0  # fixed weight; could be parameterised
        else:
            freq_weight = 0.0

        # Neighborhood generation block
        if neighborhood == "swap_adjacent_neighborhood":  # adjacent swap neighborhood
            indices = [i for i in range(n - 1) if current[i][0] != current[i + 1][0]]
            if not indices:
                logging.getLogger("jssp").info(
                    "[tabu] no moves available iter=%d best=%s evals=%d", it, best_c, evals
                )
                break
            admissible: list[tuple[float, int, tuple, list[OperationKey], int]] = []
            tabu_only: list[tuple[float, int, tuple, list[OperationKey], int]] = []
            for i in indices:
                new_perm = swap_adjacent(current, i)
                op_a, op_b = sorted((current[i], current[i + 1]))
                move_key = (op_a, op_b)
                c = evaluate(data, new_perm, cache=cache)
                evals += 1
                move_freq_pen = freq_weight * freq.get(move_key, 0)
                adjusted = c + move_freq_pen
                expiration = tabu.get(move_key, -1)
                is_tabu = move_key in tabu and expiration >= it
                if is_tabu and not (aspiration and c < best_c):
                    tabu_only.append((adjusted, c, move_key, new_perm, expiration))
                else:
                    admissible.append((adjusted, c, move_key, new_perm, expiration))
            if admissible:
                adjusted, c_raw, key_raw, perm_raw, _ = min(admissible, key=lambda x: x[0])
                best_move_c = c_raw
                best_move_key = key_raw
                best_move_perm = perm_raw
            else:
                if tabu_only:
                    # Fallback: best tabu (min adjusted then earliest expiry)
                    tabu_only.sort(key=lambda x: (x[0], x[4]))
                    adjusted, c_raw, key_raw, perm_raw, _ = tabu_only[0]
                    best_move_c = c_raw
                    best_move_key = key_raw
                    best_move_perm = perm_raw
                    used_tabu_fallback = True
                else:
                    logging.getLogger("jssp").info(
                        "[tabu] no moves (all rejected) iter=%d best=%s evals=%d",
                        it,
                        best_c,
                        evals,
                    )
                    break

        elif neighborhood == "fibonachi_neighborhood":
            new_perm, c = create_fibonachi_neighborhood(
                current,
                data=data,
                cache=cache,
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
                    # Penalty: sum frequency penalties of constituent pairs (multi-swap)
                    move_freq_pen = 0.0
                    if isinstance(move_key, tuple) and move_key and isinstance(move_key[0], tuple):
                        for pair in move_key:  # type: ignore
                            move_freq_pen += freq_weight * freq.get(pair, 0)
                    else:
                        # single pair
                        if (
                            isinstance(move_key, tuple)
                            and len(move_key) == 2
                            and not isinstance(move_key[0], tuple)
                        ):
                            move_freq_pen = freq_weight * freq.get(move_key, 0)  # type: ignore
                    adjusted = c + move_freq_pen
                    admissible_move = not (is_tabu and not (aspiration and c < best_c))
                    if admissible_move:
                        if best_move_c is None or adjusted < (best_move_c + move_freq_pen):
                            best_move_c = c
                            best_move_key = move_key
                            best_move_perm = new_perm
                    else:
                        # Fallback if entire composite move tabu set
                        if best_move_perm is None:
                            best_move_c = c
                            best_move_key = move_key
                            best_move_perm = new_perm
                            used_tabu_fallback = True
        else:
            raise ValueError(f"Unknown TABU neighborhood: {neighborhood}")

        if best_move_perm is None:
            logging.getLogger("jssp").info(
                "[tabu] no admissible move iter=%d best=%s evals=%d", it, best_c, evals
            )
            break
        current = best_move_perm
        current_c = best_move_c if best_move_c is not None else current_c
        if best_move_key is not None:
            # Update tabu + frequency memory (multi-swap: update every pair)
            expiration_iter = it + current_tenure
            tabu[best_move_key] = expiration_iter
            if (
                isinstance(best_move_key, tuple)
                and best_move_key
                and isinstance(best_move_key[0], tuple)
            ):
                for pair in best_move_key:  # type: ignore
                    if len(pair) == 2:
                        freq[pair] = freq.get(pair, 0) + 1
            else:
                if isinstance(best_move_key, tuple) and len(best_move_key) == 2:
                    freq[best_move_key] = freq.get(best_move_key, 0) + 1
        expired = [k for k, exp in tabu.items() if exp < it]
        for k in expired:
            del tabu[k]
        if current_c < best_c:
            best_c = int(current_c)
            best_perm = list(current)
            last_improvement_iter = it
            # Reset dynamic tenure after improvement
            current_tenure = base_tenure
        current_history.append(current_c)
        logging.getLogger("jssp").info(
            "[tabu] iter %d/%d current=%s best=%s evals=%d%s",
            it,
            iterations,
            current_c,
            best_c,
            evals,
            " fallback_tabu" if used_tabu_fallback else "",
        )
        if trace_file is not None:
            # format: it;current;best;move_key;evals
            try:
                with open(trace_file, "a", encoding="utf-8") as tf:
                    suffix = "FALLBACK" if used_tabu_fallback else ""
                    tf.write(f"{it};{current_c};{best_c};{best_move_key};{evals};{suffix}\n")
            except Exception:
                pass
        # Optional per-iteration Gantt snapshot
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

    # Canonical SA: random neighbor + probabilistic acceptance exp(-delta/T)
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
    # No separate best list – history of current is enough for plotting
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
        # Generate one (or several attempts to get one) neighbor
        move_repr: str | None = None
        if neighborhood == "swap_adjacent_neighborhood":
            # Random single adjacent swap (canonical SA).
            # neighbor_moves>1: several attempts seeking a real move.
            if len(current) < 2:
                neigh = current[:]
                c = current_c
            else:
                indices = [i for i in range(len(current) - 1) if current[i][0] != current[i + 1][0]]
                if not indices:
                    neigh = current[:]
                    c = current_c
                else:
                    neigh = current[:]
                    c = current_c
                    for _ in range(max(1, neighbor_moves)):
                        idx = rng.choice(indices)
                        cand = swap_adjacent(current, idx)
                        if cand != current:
                            neigh = cand
                            c = evaluate(data, neigh, cache=cache)
                            evals += 1
                            move_repr = f"swap_adj@{idx}"
                            break
                    # If no real move found we keep state
        elif neighborhood == "fibonachi_neighborhood":
            if len(current) < 2:
                neigh = current[:]
                c = current_c
            else:
                neigh, c = create_fibonachi_neighborhood(
                    current[:],
                    data=data,
                    cache=cache,
                    debug=False,
                    return_cost=True,
                )
                evals += 1
                if neigh != current:
                    # Reconstruct indices of performed swaps for tracing
                    swapped_idx = []
                    for i in range(len(current) - 1):
                        if (
                            neigh[i] == current[i + 1]
                            and neigh[i + 1] == current[i]
                            and current[i][0] != current[i + 1][0]
                        ):
                            swapped_idx.append(i)
                    if swapped_idx:
                        move_repr = "fib:" + ",".join(map(str, swapped_idx))
        else:
            raise ValueError(f"Unknown SA neighborhood: {neighborhood}")
        delta = c - current_c
        accept = False
        if delta <= 0:
            accept = True
        elif T > 0:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                accept = True
        if accept:
            current = neigh
            current_c = c
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
