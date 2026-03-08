"""Quantum Dynasearch neighborhood - select non-overlapping endpoint swaps using QUBO.

QUBO formulation (interval no-overlap constraint):
    H = Σₖ δₖ·xₖ + P·Σ_{k<l, overlap(k,l)} xₖ·xₗ

Where:
    K = number of candidate swap pairs (at most n(n-1)/2)
    xₖ ∈ {0,1} indicates whether to perform swap k = (iₖ, jₖ) with iₖ < jₖ
    δₖ = ΔCₘₐₓ after swapping endpoints of segment [iₖ..jₖ]
    P = penalty weight for overlapping intervals

Endpoint swap definition:
    Swap k = (iₖ, jₖ) exchanges πᵢₖ ↔ πⱼₖ, leaving π_{iₖ+1}..π_{jₖ-1} unchanged.
    Segment [πᵢₖ, π_{iₖ+1}, ..., π_{jₖ-1}, πⱼₖ] becomes
            [πⱼₖ, π_{iₖ+1}, ..., π_{jₖ-1}, πᵢₖ]

Overlap condition:
    Two intervals (i₁,j₁) and (i₂,j₂) overlap when max(i₁,i₂) ≤ min(j₁,j₂).

    Examples:
        [0,4] and [2,6] overlap:  max(0,2)=2 ≤ min(4,6)=4  ✓
        [0,2] and [3,5] disjoint: max(0,3)=3 > min(2,5)=2  ✗
        [0,5] and [1,3] overlap:  max(0,1)=1 ≤ min(5,3)=3  ✓ (nested)

    Note: Nesting is NOT allowed in this formulation (unlike Motzkin).
          Nested intervals overlap by the above definition.

QUBO matrix Q where H = Σₖₗ Q[k,l]·xₖ·xₗ:
    Q[k,k] = δₖ                           (linear term: cost of selecting swap k)
    Q[k,l] = P  if intervals k,l overlap  (quadratic term: penalty for conflict)
    Q[k,l] = 0  otherwise                 (non-overlapping pairs allowed)

Penalty weight:
    P = Σₖ |δₖ| + 1                       (ensures no-overlap constraint dominates)

Complexity:
    - Variables: K = O(n²) binary variables (up to n(n-1)/2 pairs)
    - Coefficients: O(K²) = O(n⁴) entries (dense conflict graph)
    - Delta computation: O(m·n³) using Head+Tail for all pairs

Optimization (classical simulation only):
    1. L_max parameter: limits segment length to reduce K from O(n²) to O(n·L_max)
    2. Filter δₖ ≥ 0: removes non-improving candidates (harmless for additive model)

    These are NOT needed on real quantum hardware (QPU handles full O(n²) natively)

Objective: Find binary assignment minimizing H, selecting non-overlapping
           endpoint swaps that improve makespan
"""

from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import (
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
    solve_qubo,
)
from src.permutation_procesing import c_max


def _intervals_overlap(i1: int, j1: int, i2: int, j2: int) -> bool:
    """Check if two intervals [i1, j1] and [i2, j2] overlap."""
    return max(i1, i2) <= min(j1, j2)


def _validate_no_overlap_intervals(
    selected: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Remove overlapping intervals, keeping non-overlapping ones greedily.

    Greedy strategy: sort by left endpoint, keep interval if it doesn't
    overlap with any previously kept interval.

    Args:
        selected: List of (i, j) interval pairs, sorted by left endpoint

    Returns:
        List of non-overlapping intervals
    """
    if not selected:
        return []

    valid: List[Tuple[int, int]] = []
    for i, j in sorted(selected):
        if not valid or i > valid[-1][1]:
            valid.append((i, j))
    return valid


def quantum_dynasearch_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 5,
    L_max: Optional[int] = None,
    backend: str = "simulator",
    dwave_token: str | None = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Quantum dynasearch neighborhood - selects non-overlapping endpoint swaps via QUBO.

    Uses Head+Tail for delta computation, then builds a QUBO where:
    - Each variable xₖ represents a candidate swap (i, j)
    - Penalty terms forbid overlapping intervals

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of samples for the QUBO solver
        L_max: Optional max interval length (j - i + 1). None = all pairs.
        backend: "simulator" or "dwave"
        dwave_token: D-Wave API token (required when backend="dwave")

    Returns:
        (new_pi, new_cmax, selected_swaps): New permutation, Cmax,
        list of (i, j) swap pairs applied
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # Compute Head+Tail matrices
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    m = len(processing_times)
    base_c = Head[m - 1][n - 1]

    # Enumerate candidates and compute deltas
    # NOTE(classical-only): L_max limits segment length to reduce variable count.
    # TODO: Remove L_max when running on a real quantum computer — QPU can handle
    #       the full O(n²) variable space natively.
    all_candidates: List[Tuple[int, int, float]] = []  # (i, j, delta)
    for i in range(n - 1):
        j_max = n - 1 if L_max is None else min(n - 1, i + L_max - 1)
        for j in range(i + 1, j_max + 1):
            delta = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            all_candidates.append((i, j, delta))

    if not all_candidates:
        return pi.copy(), base_c, []

    # NOTE(classical-only): Filter out non-improving candidates (δ ≥ 0) to reduce
    # QUBO size for SimulatedAnnealingSampler. This is mathematically equivalent
    # within the additive-delta QUBO model — the solver would set these x_k = 0
    # anyway, since δ_k ≥ 0 never decreases H.
    # TODO: Remove this filter when running on a real quantum computer — QPU
    #       handles larger variable counts and the filter is unnecessary.
    candidates = [(i, j, d) for i, j, d in all_candidates if d < 0]

    # If no improving candidate, fallback to single best (even if δ ≥ 0)
    if not candidates:
        best = min(all_candidates, key=lambda x: x[2])
        if best[2] < 0:
            new_pi = pi.copy()
            new_pi[best[0]], new_pi[best[1]] = new_pi[best[1]], new_pi[best[0]]
            return new_pi, c_max(new_pi, processing_times), [(best[0], best[1])]
        return pi.copy(), base_c, []

    num_vars = len(candidates)

    # Build QUBO: interval no-overlap
    penalty = sum(abs(d) for _, _, d in candidates) + 1
    Q: Dict[Tuple[str, str], float] = {}

    # Diagonal: cost of selecting each swap
    for k in range(num_vars):
        Q[(f"x{k}", f"x{k}")] = candidates[k][2]  # delta_k

    # Off-diagonal: penalty for overlapping intervals
    for k in range(num_vars):
        i1, j1, _ = candidates[k]
        for l in range(k + 1, num_vars):
            i2, j2, _ = candidates[l]
            if _intervals_overlap(i1, j1, i2, j2):
                Q[(f"x{k}", f"x{l}")] = penalty

    # Solve QUBO
    solution = solve_qubo(Q, num_reads, backend=backend, dwave_token=dwave_token)
    selected_indices = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # Extract selected intervals and validate no-overlap
    selected_intervals = [(candidates[k][0], candidates[k][1]) for k in selected_indices]
    valid_swaps = _validate_no_overlap_intervals(selected_intervals)

    # Fallback: if nothing selected, pick the single swap with minimal delta
    if not valid_swaps:
        best_k = min(range(num_vars), key=lambda k: candidates[k][2])
        if candidates[best_k][2] < 0:
            valid_swaps = [(candidates[best_k][0], candidates[best_k][1])]

    # Apply chosen swaps
    new_pi = pi.copy()
    for i, j in sorted(valid_swaps):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps
