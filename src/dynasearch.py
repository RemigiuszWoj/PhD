import copy
from typing import Dict, Iterator, List, Optional, Tuple

from src.permutation_procesing import c_max


# --------------------------
# Prefix boundaries (F)
# --------------------------
def compute_prefix_boundaries(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """
    Compute prefix boundary vectors F[k] for k = 0..n.
    F[k] is a length-m vector: completion times on machines after processing
    first k jobs (pi[0..k-1]).
    """
    m = len(processing_times)
    n = len(pi)
    F = [[0] * m for _ in range(n + 1)]
    for k in range(1, n + 1):
        job = pi[k - 1]
        F[k][0] = F[k - 1][0] + processing_times[0][job]
        for r in range(1, m):
            F[k][r] = max(F[k][r - 1], F[k - 1][r]) + processing_times[r][job]
    return F  # F[0]..F[n]; F[n][-1] == c_max(pi)


# --------------------------
# Suffix boundaries (optional)
# --------------------------
def compute_suffix_boundaries(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """
    Compute suffix boundary vectors S[k] for k = 0..n where S[k] describes
    completion times if we process suffix pi[k..n-1] on 'reversed' machines.
    This function returns vectors such that you can reason about the tail,
    but in our implementation we will usually continue forward from boundary,
    so S is optional and provided for completeness / potential optimizations.
    """
    # We'll compute by reversing the instance: reverse job order and machine order,
    # compute prefixes there, then map back. This yields information about tails.
    m = len(processing_times)
    n = len(pi)
    # Build reversed processing times for reversed job order and reversed machines
    # processing_times_rev[r_rev][job_index] = processing_times[m-1-r_rev][ pi[n-1-job_index] ]
    # But for now produce simple suffix by computing completions of tails as if they
    # were independent prefixes: compute completion vectors for suffixes directly by simulation.
    S = [
        [0] * m for _ in range(n + 1)
    ]  # S[k] will be completion after processing jobs k..n-1 starting from zero
    # compute for each k the completion vector if we start from zero and process pi[k..]
    for k in range(n - 1, -1, -1):
        # simulate processing of suffix pi[k..] from zero start
        col_prev = [0] * m
        for t in range(k, n):
            j = pi[t]
            col = [0] * m
            col[0] = col_prev[0] + processing_times[0][j]
            for r in range(1, m):
                col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][j]
            col_prev = col
        S[k] = col_prev[:]  # completion vector after processing suffix
    # S[n] is zeros (empty suffix)
    return S


# --------------------------
# compute from prefix boundary, apply local block, then continue with tail
# --------------------------
def compute_from_boundary_and_continue(
    boundary: List[int],
    local_seq: List[int],
    tail_jobs: List[int],
    processing_times: List[List[int]],
) -> Tuple[int, List[int]]:
    """
    Start from 'boundary' (vector length m), process local_seq in order,
    then process tail_jobs in order. Return (final_Cmax, last_col_after_local_seq).
    Uses same recurrence as c_max but starts from boundary for column before local_seq.
    """
    m = len(processing_times)
    col_prev = boundary[:]  # column at position before local segment
    # process local sequence
    for job in local_seq:
        col = [0] * m
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
        col_prev = col
    last_col_after_local = col_prev[:]
    # continue with tail
    for job in tail_jobs:
        col = [0] * m
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
        col_prev = col
    final_cmax = col_prev[-1]
    return final_cmax, last_col_after_local


# --------------------------
# Core: Dynasearch (article-style, prefix-assisted)
# --------------------------
def dynasearch_full(
    pi: List[int],
    processing_times: List[List[int]],
    L_max: Optional[int] = None,
    force_move_if_none: bool = True,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """
    Dynasearch neighborhood faithful to article structure (practical implementation).

    Returns:
        new_pi, final_cmax, chosen_moves_list
    chosen_moves_list is list of (i,j) intervals applied (sorted by i).

    Parameters:
      - L_max: optional limit on maximum interval length (j - i + 1). If None, consider all pairs.
      - force_move_if_none: if True, when DP picks nothing, force smallest-delta move.
    """
    n = len(pi)
    if n < 2:
        return pi, c_max(pi, processing_times), []

    # prefix boundaries (F[k]) to avoid recomputing left part for each candidate
    F = compute_prefix_boundaries(pi, processing_times)
    base_c = F[n][-1]  # c_max original

    candidates: List[Tuple[int, int, int]] = []  # (i, j, delta)
    # enumerate candidate 2-exchanges (i,j) where i<j
    for i in range(0, n - 1):
        # apply L_max constraint on interval length if provided
        j_max = n - 1 if L_max is None else min(n - 1, i + L_max - 1)
        for j in range(i + 1, j_max + 1):
            # build local sequence for positions i..j after performing the 2-exchange
            # Here we consider the simple 2-exchange swapping endpoints i and j.
            local_seq = pi[i : j + 1].copy()
            # apply swap endpoints (move pi[j] to position i, pi[i] to position j)
            local_seq[0], local_seq[-1] = local_seq[-1], local_seq[0]
            # tail jobs after j
            tail_jobs = pi[j + 1 :] if j + 1 < n else []
            # compute final Cmax by starting from prefix boundary F[i]
            final_c, _ = compute_from_boundary_and_continue(
                F[i], local_seq, tail_jobs, processing_times
            )
            delta = final_c - base_c
            candidates.append((i, j, delta))

    if not candidates:
        return pi, base_c, []

    # sort candidates by left index then by right index (stable)
    candidates.sort(key=lambda x: (x[0], x[1]))
    m = len(candidates)
    memo: Dict[int, Tuple[int, Tuple[Tuple[int, int], ...]]] = {}

    # DP: choose subset of non-overlapping intervals minimizing total delta
    def solve(pos: int) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        if pos >= m:
            return 0, ()
        if pos in memo:
            return memo[pos]

        i, j, delta = candidates[pos]
        # option skip
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set

        # option take -> find next pos that doesn't overlap with [i,j]
        next_pos = pos + 1
        while next_pos < m:
            ii, jj, _ = candidates[next_pos]
            # overlap if intervals share any index
            if not (jj < i or ii > j):
                next_pos += 1
            else:
                break
        take_rest_val, take_rest_set = solve(next_pos)
        take_val = delta + take_rest_val
        if take_val < best_val:
            best_val = take_val
            best_set = ((i, j),) + take_rest_set

        memo[pos] = (best_val, best_set)
        return memo[pos]

    total_delta, chosen = solve(0)

    # if DP picked nothing and we want to force a move => choose single minimal delta move
    chosen_intervals: List[Tuple[int, int]] = list(chosen)
    if not chosen_intervals and force_move_if_none:
        best_i, best_j, _ = min(candidates, key=lambda x: x[2])
        chosen_intervals = [(best_i, best_j)]

    # apply chosen swaps on a copy of pi (sorted by left index)
    new_pi = pi.copy()
    for i, j in sorted(chosen_intervals, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    final_c = c_max(new_pi, processing_times)
    return new_pi, final_c, chosen_intervals
