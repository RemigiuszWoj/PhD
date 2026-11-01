"""k-bonacci neighborhood family - disjoint endpoint swaps of bounded distance.

A compound move is a set of endpoint swaps (i, j) on pairwise disjoint
intervals [i, j] with swap distance d = j - i in {1, ..., k-1}; the jobs
strictly inside an interval stay in place.

    k = 2  -> Fibonacci   (adjacent swaps only, |N| = F(n+1))
    k = 3  -> tribonacci  (distances 1 and 2)
    k      -> k-bonacci   (distances 1 .. k-1)
    k = n  -> Dynasearch  (any distance, |N| = 2^(n-1))

Counting: a compound move is a tiling of the n positions by tiles of
length 1 (no move) or L = d + 1 in {2, ..., k} (swap the tile endpoints),
hence |N_k(n)| = a_n with a_n = a_{n-1} + ... + a_{n-k} (k-bonacci numbers).

Evaluation:
    deltas  : Head+Tail, O(m * d) per candidate -> O(m * n * k^2) total
    filter  : block-boundary filter of the previous papers (last-machine test,
              ``compute_block_boundaries``); the "_cp" ablation uses the exact
              critical-path boundaries (lossless), ``compute_critical_path_boundaries``
    DP      : positional DP over prefixes, O(n * k)

The DP minimizes the sum of the individual deltas (i.e. maximizes the
predicted improvement). In the PFSP the makespan
change of a compound move is NOT additive, so the predicted delta can
differ from the real one; ``kbonacci_neighborhood`` reports both so that
the additivity error can be studied as a function of k.
"""


from src.neighborhoods.accelerator import (
    compute_block_boundaries,
    compute_critical_path_boundaries,
)
from src.neighborhoods.common import compute_endpoint_swap_delta, compute_head, compute_tail
from src.permutation_procesing import c_max

Candidate = tuple[int, int, int]  # (i, j, delta)


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------
def count_kbonacci_moves(n: int, k: int, include_identity: bool = True) -> int:
    """Number of compound moves |N_k(n)| (k-bonacci recurrence).

    Args:
        n: permutation length
        k: maximum tile length (swap distance <= k-1); k >= 2
        include_identity: count the empty move (no swap) as well
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    a = [1] + [0] * n
    for p in range(1, n + 1):
        a[p] = sum(a[p - L] for L in range(1, min(k, p) + 1))
    return a[n] if include_identity else a[n] - 1


def growth_constant(k: int, tol: float = 1e-12) -> float:
    """Largest real root of x^k = x^(k-1) + ... + 1 (k-bonacci constant)."""
    lo, hi = 1.0, 2.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if mid**k - sum(mid**e for e in range(k)) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def compute_candidates(
    pi: list[int],
    processing_times: list[list[int]],
    k: int,
    use_block_accelerator: bool = True,
    boundary_mode: str = "last_machine",
) -> tuple[int, list[Candidate]]:
    """All endpoint swaps (i, j) with 1 <= j - i <= k - 1 and their deltas.

    Returns:
        (base_cmax, candidates) with candidates = [(i, j, delta), ...]
    """
    n = len(pi)
    m = len(processing_times)
    if n < 2:
        return c_max(pi, processing_times), []

    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    base_c = Head[m - 1][n - 1]

    boundaries: list[int] | None = None
    if use_block_accelerator:
        if boundary_mode == "critical_path":
            boundaries = compute_critical_path_boundaries(Head, processing_times, pi)
        else:  # "last_machine": the test used by all neighborhoods of the previous papers
            boundaries = compute_block_boundaries(Head, processing_times, pi)
    is_boundary = [False] * n
    if boundaries is not None:
        for u in boundaries:
            is_boundary[u] = True

    d_max = min(k - 1, n - 1)
    candidates: list[Candidate] = []
    all_candidates: list[Candidate] = []
    for i in range(n - 1):
        # running flag: does [i, j] contain a block boundary?
        has_boundary = is_boundary[i]
        for j in range(i + 1, min(n - 1, i + d_max) + 1):
            has_boundary = has_boundary or is_boundary[j]
            if boundaries is not None and not has_boundary:
                all_candidates.append((i, j, 0))  # placeholder, evaluated lazily below
                continue
            delta = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            candidates.append((i, j, delta))

    if boundaries is not None and not candidates:
        # everything blocked (rare) -> evaluate the full set, like the NPI fallback
        candidates = [
            (i, j, compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c))
            for i, j, _ in all_candidates
        ]
    return base_c, candidates


# ---------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------
def solve_dp(n: int, candidates: list[Candidate]) -> tuple[int, list[tuple[int, int]]]:
    """Best set of pairwise disjoint intervals minimizing the sum of deltas.

    best[p] = optimum over positions 0..p-1:
        best[p] = min( best[p-1],                       # position p-1 untouched
                       min_{(i, p-1, d)} best[i] + d )   # interval ending at p-1

    Complexity O(n + |candidates|) = O(n * k).
    """
    ending_at: dict[int, list[Candidate]] = {}
    for c in candidates:
        ending_at.setdefault(c[1], []).append(c)

    best = [0] * (n + 1)
    choice: list[Candidate | None] = [None] * (n + 1)
    for p in range(1, n + 1):
        best[p] = best[p - 1]
        for c in ending_at.get(p - 1, ()):
            val = best[c[0]] + c[2]
            if val < best[p]:
                best[p] = val
                choice[p] = c

    chosen: list[tuple[int, int]] = []
    p = n
    while p > 0:
        c = choice[p]
        if c is None:
            p -= 1
        else:
            chosen.append((c[0], c[1]))
            p = c[0]
    chosen.reverse()
    return best[n], chosen


def apply_endpoint_swaps(pi: list[int], swaps: list[tuple[int, int]]) -> list[int]:
    new_pi = pi.copy()
    for i, j in swaps:
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    return new_pi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def kbonacci_neighborhood(
    pi: list[int],
    processing_times: list[list[int]],
    k: int,
    use_block_accelerator: bool = True,
    force_move_if_none: bool = True,
    stats: dict | None = None,
    boundary_mode: str = "last_machine",
) -> tuple[list[int], int, list[tuple[int, int]]]:
    """Best compound move of the k-bonacci neighborhood.

    Args:
        pi: current permutation
        processing_times: m x n matrix
        k: tile length bound (swap distance <= k-1); k >= n gives Dynasearch
        use_block_accelerator: block-boundary candidate filter (as in the previous papers)
        force_move_if_none: if the DP selects nothing, apply the best single swap
        stats: optional dict filled with diagnostics (predicted/real delta,
            number of swaps, swap distances, candidate count)

    Returns:
        (new_pi, new_cmax, chosen_swaps)
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    n = len(pi)
    base_c, candidates = compute_candidates(
        pi, processing_times, k, use_block_accelerator, boundary_mode
    )
    if not candidates:
        return pi.copy(), base_c, []

    predicted, chosen = solve_dp(n, candidates)
    if not chosen and force_move_if_none:
        bi, bj, predicted = min(candidates, key=lambda c: c[2])
        chosen = [(bi, bj)]

    new_pi = apply_endpoint_swaps(pi, chosen)
    new_c = c_max(new_pi, processing_times)

    if stats is not None:
        stats.update(
            {
                "k": k,
                "base_cmax": base_c,
                "num_candidates": len(candidates),
                "predicted_delta": predicted,
                "real_delta": new_c - base_c,
                "num_swaps": len(chosen),
                "distances": [j - i for i, j in chosen],
            }
        )
    return new_pi, new_c, chosen


def parse_kbonacci_mode(mode: str, n: int) -> int | None:
    """Map a neighborhood mode name to k, or None if it is not a k-bonacci mode.

    Accepted: "tribonacci" (k=3), "tetranacci" (k=4), "kbonacci_k<K>",
    "kbonacci_kn" (k=n, i.e. Dynasearch). A trailing "_cp" (ablation: exact
    critical-path filter) is ignored here; see ``kbonacci_boundary_mode``.
    """
    if mode.endswith("_cp"):
        mode = mode[: -len("_cp")]
    if mode == "tribonacci":
        return 3
    if mode == "tetranacci":
        return 4
    if mode.startswith("kbonacci_k"):
        tail = mode[len("kbonacci_k") :]
        if tail == "n":
            return max(2, n)
        if tail.isdigit() and int(tail) >= 2:
            return int(tail)
    return None


def kbonacci_boundary_mode(mode: str) -> str:
    """'critical_path' for the "_cp" ablation modes, else 'last_machine' (previous papers)."""
    return "critical_path" if mode.endswith("_cp") else "last_machine"
