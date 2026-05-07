"""Accelerators for PFSP neighborhood search.

Based on: Smutnicki, C. "Algorytmy szeregowania", rozdział 7.9.

Implemented accelerators:
    compute_block_boundaries  - find block structure of permutation
    is_blocked_adjacent_swap  - block property for Adjacent/Fibonacci
    filter_blocked_swaps      - remove blocked swaps from candidate list

Block property (właściwość blokowa):
    Given the block decomposition u_0=0 < u_1 < ... < u_k = n of
    permutation π, an adjacent swap v=(a, a+1) where both positions
    lie strictly inside the same block (u_{i-1} < a < a+1 < u_i)
    cannot improve C_max(π). Such swaps can be safely skipped.

    This reduces the effective neighborhood size from n-1 to at most
    2(m-1) promising swaps (those crossing block boundaries), giving
    an O(n/m) speedup on average.
"""

from typing import List, Tuple


def compute_block_boundaries(
    Head: List[List[int]],
    processing_times: List[List[int]],
    pi: List[int],
) -> List[int]:
    """Compute block boundary positions u_i for permutation π.

    A position j is a block boundary if the last machine does not wait
    for the previous machine at position j, i.e., the last machine is
    the bottleneck:

        Head[m-1][j] == Head[m-2][j] + p_{m, π(j)}

    which means Head[m-1][j] > Head[m-1][j-1] + p_{m-1, π(j)}
    (last machine doesn't wait).

    Equivalently using the standard formulation:
        Head[m-1][j] = Head[m-2][j] + p[m-1][pi[j]]
        (i.e., Head[m-1][j-1] <= Head[m-2][j])

    Args:
        Head: Head matrix (forward completion times), size m×n
        processing_times: m × n processing times matrix
        pi: Current permutation

    Returns:
        Sorted list of block boundary positions [u_1, u_2, ..., u_k=n-1]
        where u_0 = -1 (implicit left boundary before position 0).

        A swap at position a is inside block i iff u_{i-1} < a < u_i.
    """
    m = len(processing_times)
    n = len(pi)

    if m < 2:
        # Single machine: no block structure, all swaps potentially useful
        return list(range(n))

    boundaries = []
    for j in range(n):
        # Position j is a boundary if last machine is NOT waiting:
        # Head[m-1][j] is determined by Head[m-2][j], not Head[m-1][j-1]
        head_from_above = Head[m - 2][j] + processing_times[m - 1][pi[j]]
        head_from_left = (Head[m - 1][j - 1] if j > 0 else 0) + processing_times[m - 1][pi[j]]
        # Boundary: last machine starts immediately after machine m-2 finishes
        # i.e., it does NOT wait for the previous position on machine m-1
        if Head[m - 1][j] == head_from_above:
            boundaries.append(j)

    # Always include n-1 as final boundary
    if not boundaries or boundaries[-1] != n - 1:
        boundaries.append(n - 1)

    return boundaries


def is_blocked_adjacent_swap(
    a: int,
    boundaries: List[int],
) -> bool:
    """Check if adjacent swap at position a is blocked (cannot improve Cmax).

    Swap v=(a, a+1) is blocked if both positions a and a+1 lie strictly
    inside the same block, i.e., there is no boundary u_i with a <= u_i <= a+1.

    Equivalently: swap is NOT blocked iff it crosses or touches a boundary.

    Args:
        a: Swap position (swaps elements at positions a and a+1)
        boundaries: Sorted list of block boundary positions

    Returns:
        True if swap is blocked (can be safely skipped),
        False if swap may improve Cmax (should be evaluated)
    """
    # Swap (a, a+1) crosses boundary u_i iff a < u_i <= a+1
    # i.e., u_i == a+1 or u_i == a (boundary at either endpoint)
    # More precisely: swap is useful only if a boundary lies at a or a+1
    for u in boundaries:
        if u == a or u == a + 1:
            return False  # touches boundary → not blocked
        if a < u < a + 1:
            return False  # impossible for integers but kept for clarity
    return True  # no boundary near swap → blocked


def filter_blocked_swaps(
    candidates: List[Tuple[int, float]],
    boundaries: List[int],
) -> List[Tuple[int, float]]:
    """Remove blocked swaps from candidate list.

    Args:
        candidates: List of (position, delta) pairs
        boundaries: Block boundary positions from compute_block_boundaries

    Returns:
        Filtered list with blocked swaps removed.
        If ALL swaps are blocked (rare), returns original list unfiltered
        to ensure algorithm always makes progress.
    """
    filtered = [
        (pos, delta) for pos, delta in candidates if not is_blocked_adjacent_swap(pos, boundaries)
    ]
    # Safety fallback: if everything is blocked, return original
    return filtered if filtered else candidates


def compute_block_boundaries_from_pi(
    pi: List[int],
    processing_times: List[List[int]],
) -> List[int]:
    """Convenience wrapper: compute boundaries directly from permutation.

    Computes Head matrix internally. Use when Head is not already available.
    For efficiency, prefer compute_block_boundaries when Head is already
    computed (e.g., inside adjacent/fibonacci neighborhood evaluation).

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix

    Returns:
        Block boundary positions (see compute_block_boundaries)
    """
    from src.neighborhoods.common import compute_head

    Head = compute_head(pi, processing_times)
    return compute_block_boundaries(Head, processing_times, pi)


def count_useful_swaps(boundaries: List[int], n: int) -> int:
    """Count non-blocked adjacent swaps given block boundaries.

    Useful for estimating speedup from block property.
    At most 2*(len(boundaries)-1) swaps cross boundaries.

    Args:
        boundaries: Block boundary positions
        n: Permutation length

    Returns:
        Number of non-blocked swaps
    """
    useful = 0
    for u in boundaries:
        # Swaps at u-1 and u cross or touch boundary u
        if u - 1 >= 0:
            useful += 1
        if u < n - 1:
            useful += 1
    return min(useful, n - 1)  # cap at total number of possible swaps


# ---------------------------------------------------------------------------
# Shared helpers for Dynasearch and Motzkin neighborhoods
# ---------------------------------------------------------------------------

def intervals_overlap(i1: int, j1: int, i2: int, j2: int) -> bool:
    """True if intervals [i1,j1] and [i2,j2] share at least one index."""
    return max(i1, i2) <= min(j1, j2)


def motzkin_conflict(i1: int, j1: int, i2: int, j2: int) -> bool:
    """True if pairs (i1,j1) and (i2,j2) conflict under Motzkin rules.

    Conflicts: shared endpoint OR crossing arcs.
    Allowed: disjoint arcs OR nested arcs (i1 < i2 < j2 < j1).
    """
    if {i1, j1} & {i2, j2}:
        return True
    if i1 < i2 < j1 < j2:
        return True
    if i2 < i1 < j2 < j1:
        return True
    return False


def filter_blocked_pairs_npi(
    candidates: List[Tuple[int, float]],
    boundaries: List[int],
) -> List[Tuple[int, float]]:
    """Filter endpoint-swap pairs using the NPI block property.

    Pair (i, j, ...) is promising only if at least one block boundary u
    lies within [i, j]. Pairs whose span contains no boundary cannot
    improve Cmax and are skipped.

    Works with both 2-tuples (i, j) and 3-tuples (i, j, delta).

    Args:
        candidates: List of (i, j) or (i, j, delta) tuples
        boundaries: Block boundary positions from compute_block_boundaries

    Returns:
        Filtered list. Falls back to original if all pairs are blocked.
    """
    boundary_set = set(boundaries)
    filtered = [
        c for c in candidates
        if any(c[0] <= u <= c[1] for u in boundary_set)
    ]
    return filtered if filtered else candidates


def validate_no_overlap_intervals(
    selected: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Greedy feasibility repair for Dynasearch: remove overlapping intervals.

    Sorts by left endpoint and keeps each interval only if it doesn't
    overlap with the last accepted one.

    Args:
        selected: List of (i, j) interval pairs

    Returns:
        Non-overlapping subset (greedy, left-to-right)
    """
    valid: List[Tuple[int, int]] = []
    for i, j in sorted(selected):
        if not valid or i > valid[-1][1]:
            valid.append((i, j))
    return valid


def validate_motzkin_selection(
    selected: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Greedy feasibility repair for Motzkin: remove conflicting pairs.

    Iterates over pairs sorted by left endpoint, keeping each pair only
    if it doesn't conflict (cross or share endpoint) with any already
    accepted pair.

    Args:
        selected: List of (i, j) pairs chosen by the QUBO solver

    Returns:
        Motzkin-admissible subset
    """
    valid: List[Tuple[int, int]] = []
    for pair in sorted(selected):
        if all(not motzkin_conflict(pair[0], pair[1], v[0], v[1]) for v in valid):
            valid.append(pair)
    return valid
