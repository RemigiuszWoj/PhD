"""Fibonacci neighborhood - non-overlapping adjacent swaps.

Selects the optimal set of non-overlapping adjacent swaps
minimizing the total Cmax change.

The name comes from the Fibonacci sequence: the number of possible sets
of non-overlapping swaps from n-1 positions equals F_{n+1}.

Method: Dynamic programming O(n)
Complexity: O(m·n) for deltas + O(n·k) for DP = O(m·n) total

Accelerator: block property (Smutnicki 7.9) optionally filters swaps that
cannot improve Cmax before the DP selection step.
"""

from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import apply_swaps, compute_head, compute_tail
from src.neighborhoods.accelerator import (
    compute_block_boundaries,
    filter_blocked_swaps,
)
from src.permutation_procesing import c_max


def _compute_deltas_fast(
    pi: List[int],
    processing_times: List[List[int]],
    use_block_accelerator: bool = True,
) -> Tuple[int, List[Tuple[int, float]]]:
    """Compute deltas for all adjacent swaps using Head+Tail technique.

    Complexity: O(m·n) instead of O(n²·m).
    With block accelerator: skips dominated swaps, reducing candidates
    to at most 2*(m-1) on average.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        use_block_accelerator: If True, skip blocked swaps (Smutnicki 7.9)

    Returns:
        (base_cmax, candidates) where candidates = [(position, delta), ...]
    """
    m = len(processing_times)
    n = len(pi)

    if n < 2:
        return c_max(pi, processing_times), []

    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)

    base_cmax = Head[m - 1][n - 1]

    # Compute block boundaries once if accelerator enabled
    boundaries: Optional[List[int]] = None
    if use_block_accelerator:
        boundaries = compute_block_boundaries(Head, processing_times, pi)

    candidates: List[Tuple[int, float]] = []

    for j in range(n - 1):
        # Block property: skip swap if both positions inside same block
        if boundaries is not None and _is_blocked(j, boundaries):
            continue

        job_a = pi[j]
        job_b = pi[j + 1]

        C_j = [0] * m
        C_j1 = [0] * m

        for i in range(m):
            left = Head[i][j - 1] if j > 0 else 0
            top = C_j[i - 1] if i > 0 else 0
            C_j[i] = max(top, left) + processing_times[i][job_b]

            top_j1 = C_j[i]
            if i == 0:
                C_j1[i] = C_j[i] + processing_times[i][job_a]
            else:
                C_j1[i] = max(C_j1[i - 1], C_j[i]) + processing_times[i][job_a]

        if j + 2 < n:
            new_cmax = max(C_j1[i] + Tail[i][j + 2] for i in range(m))
        else:
            new_cmax = C_j1[m - 1]

        delta = new_cmax - base_cmax
        candidates.append((j, delta))

    return base_cmax, candidates


def _is_blocked(a: int, boundaries: List[int]) -> bool:
    """Inline version of is_blocked_adjacent_swap for performance."""
    for u in boundaries:
        if u == a or u == a + 1:
            return False
    return True


def _solve_dp_topk(
    candidates: List[Tuple[int, float]],
    k: int,
) -> List[Tuple[float, Tuple[int, ...]]]:
    """Solve DP to find top-k best non-overlapping swap sets.

    Uses dynamic programming with memoization.

    Recurrence:
        dp[pos] = top-k of:
            - dp[pos+1]  (skip swap at pos)
            - delta[pos] + dp[pos+2]  (take swap, skip next)

    Complexity: O(n * k) time and space.

    Returns:
        List of (total_delta, chosen_positions) sorted by delta ascending
    """
    if not candidates:
        return [(0.0, ())]

    m = len(candidates)
    dp: Dict[int, List[Tuple[float, Tuple[int, ...]]]] = {}

    def solve(pos: int) -> List[Tuple[float, Tuple[int, ...]]]:
        if pos >= m:
            return [(0.0, ())]
        if pos in dp:
            return dp[pos]

        idx, delta = candidates[pos]

        # Option 1: skip this swap
        skip_solutions = solve(pos + 1)

        # Option 2: take this swap — skip next overlapping position
        next_pos = pos + 1
        while next_pos < m and candidates[next_pos][0] == idx + 1:
            next_pos += 1

        take_rest = solve(next_pos)
        take_solutions = [
            (delta + rest_delta, (idx,) + rest_set) for rest_delta, rest_set in take_rest
        ]

        all_solutions = skip_solutions + take_solutions
        all_solutions.sort(key=lambda x: x[0])

        seen = set()
        unique_topk = []
        for val, s in all_solutions:
            if s not in seen:
                seen.add(s)
                unique_topk.append((val, s))
                if len(unique_topk) >= k:
                    break

        dp[pos] = unique_topk
        return dp[pos]

    return solve(0)


def fibonacci_neighborhood_topk(
    pi: List[int],
    processing_times: List[List[int]],
    k: int,
    use_block_accelerator: bool = True,
) -> List[dict]:
    """Find top-k best sets of non-overlapping swaps.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        k: Number of top solutions to return
        use_block_accelerator: If True, apply block property filter
            before DP selection (Smutnicki 7.9 accelerator).

    Returns:
        List of dicts [{"pi": [...], "cmax": int, "move": (positions...)}, ...]
        sorted by cmax ascending
    """
    n = len(pi)
    if n < 2:
        c = c_max(pi, processing_times)
        return [{"pi": pi.copy(), "cmax": c, "move": ()}]

    base_c, candidates = _compute_deltas_fast(
        pi,
        processing_times,
        use_block_accelerator=use_block_accelerator,
    )

    if not candidates:
        return [{"pi": pi.copy(), "cmax": base_c, "move": ()}]

    topk_solutions = _solve_dp_topk(candidates, k)

    results = []
    for total_delta, chosen in topk_solutions:
        if not chosen:
            continue
        new_pi = apply_swaps(pi, list(chosen))
        final_c = c_max(new_pi, processing_times)
        results.append({"pi": new_pi, "cmax": final_c, "move": chosen})

    results.sort(key=lambda x: x["cmax"])
    return results[:k]
