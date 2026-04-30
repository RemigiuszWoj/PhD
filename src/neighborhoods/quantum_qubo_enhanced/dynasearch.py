"""Quantum Dynasearch Enhanced neighborhood - windowed QUBO for large n.

Extends quantum_qubo/dynasearch.py with a windowed decomposition strategy
that enables execution on D-Wave for large n where the full QUBO
(K = O(n²) variables) would exceed QPU capacity.

Windowed decomposition:
    The permutation π of length n is split into overlapping windows of
    size `window_size`. For each window, a separate QUBO is built and
    solved independently. Results are merged with conflict resolution.

    Window overlap (50% by default) ensures that swaps crossing window
    boundaries are not permanently excluded — they appear in adjacent
    windows. This introduces suboptimality vs. the full QUBO but is
    necessary for large n.

Feasibility (D-Wave Advantage, Lmax per window):
    Full QUBO (no window):  n=20  → K=190  ✅  n=50 → K=1225 ❌
    Windowed (window=20):   any n → K≤190  ✅  (window dominates)
    Auto window:            computed from n to keep K ≤ 180 variables

Auto window size selection:
    Given D-Wave effective capacity C_eff = 180 variables (conservative),
    and K(w) = w*(w-1)/2 for a window of size w,
    we solve K(w) ≤ C_eff → w ≤ floor((1 + sqrt(1 + 8*C_eff)) / 2).
    For C_eff=180: w_max = 19 → default window_size = 19.
"""

import math
from typing import Dict, List, Optional, Tuple

from src.neighborhoods.common import (
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
    solve_qubo,
)
from src.permutation_procesing import c_max

# Conservative D-Wave effective variable capacity for dense QUBO
_DWAVE_EFFECTIVE_CAPACITY = 180


def _auto_window_size(n: int, capacity: int = _DWAVE_EFFECTIVE_CAPACITY) -> int:
    """Compute maximum window size such that K = w*(w-1)/2 ≤ capacity."""
    w = int((1 + math.sqrt(1 + 8 * capacity)) / 2)
    return max(4, min(w, n))


def _intervals_overlap(i1: int, j1: int, i2: int, j2: int) -> bool:
    return max(i1, i2) <= min(j1, j2)


def _validate_no_overlap_intervals(
    selected: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Greedy removal of overlapping intervals (sort by left endpoint)."""
    valid: List[Tuple[int, int]] = []
    for i, j in sorted(selected):
        if not valid or i > valid[-1][1]:
            valid.append((i, j))
    return valid


def _solve_window_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    start: int,
    end: int,
    Head: List[List[int]],
    Tail: List[List[int]],
    base_c: int,
    num_reads: int,
    backend: str,
    dwave_token: str | None,
    solver: str | None,
    annealing_time_us: int | None,
    chain_strength: float | None,
    num_spin_reversal_transforms: int | None,
) -> List[Tuple[int, int]]:
    """Build and solve QUBO for window [start..end-1]. Returns global swap indices."""
    m = len(processing_times)
    candidates: List[Tuple[int, int, float]] = []
    for i in range(start, end - 1):
        for j in range(i + 1, end):
            delta = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
            candidates.append((i, j, delta))

    if not candidates:
        return []

    num_vars = len(candidates)
    penalty = sum(abs(d) for _, _, d in candidates) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for k in range(num_vars):
        Q[(f"x{k}", f"x{k}")] = candidates[k][2]
    for k in range(num_vars):
        i1, j1, _ = candidates[k]
        for l in range(k + 1, num_vars):
            i2, j2, _ = candidates[l]
            if _intervals_overlap(i1, j1, i2, j2):
                Q[(f"x{k}", f"x{l}")] = penalty

    solution = solve_qubo(
        Q,
        num_reads=num_reads,
        backend=backend,
        dwave_token=dwave_token,
        solver=solver,
        annealing_time_us=annealing_time_us,
        chain_strength=chain_strength,
        num_spin_reversal_transforms=num_spin_reversal_transforms,
    )
    selected_k = [int(v[1:]) for v, val in solution.items() if val == 1]
    return [(candidates[k][0], candidates[k][1]) for k in selected_k]


def quantum_dynasearch_enhanced(
    pi: List[int],
    processing_times: List[List[int]],
    window_size: Optional[int] = None,
    overlap_ratio: float = 0.5,
    num_reads: int = 100,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
) -> Tuple[List[int], int, List[Tuple[int, int]]]:
    """Quantum Dynasearch Enhanced — windowed QUBO decomposition for large n.

    For small n (n ≤ window_size), behaves identically to quantum_qubo/dynasearch.py
    (single QUBO, full variable space). For large n, decomposes into overlapping
    windows and merges results with greedy conflict resolution.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        window_size: Window size (number of positions per QUBO subproblem).
                     None = auto-select based on D-Wave capacity (~19).
        overlap_ratio: Fraction of window_size used as overlap between windows.
                       0.5 = 50% overlap (default). Higher = fewer missed cross-boundary swaps.
        num_reads: Number of QPU samples per window
        backend: 'simulator' or 'dwave'
        dwave_token: D-Wave API token (required when backend='dwave')
        solver: D-Wave solver name
        annealing_time_us: Annealing time in microseconds
        chain_strength: Chain strength override
        num_spin_reversal_transforms: Spin reversal transforms

    Returns:
        (new_pi, new_cmax, applied_swaps): New permutation, Cmax,
        list of (i, j) endpoint swap pairs applied
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # Determine window size
    w = window_size if window_size is not None else _auto_window_size(n)
    w = min(w, n)  # cap at n

    # Precompute Head and Tail once for full permutation
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    m_machines = len(processing_times)
    base_c = Head[m_machines - 1][n - 1]

    # Collect swaps from all windows
    all_swaps: List[Tuple[int, int]] = []
    step = max(1, int(w * (1 - overlap_ratio)))

    for start in range(0, n, step):
        end = min(start + w, n)
        if end - start < 2:
            break
        window_swaps = _solve_window_qubo(
            pi,
            processing_times,
            start,
            end,
            Head,
            Tail,
            base_c,
            num_reads,
            backend,
            dwave_token,
            solver,
            annealing_time_us,
            chain_strength,
            num_spin_reversal_transforms,
        )
        all_swaps.extend(window_swaps)
        if end == n:
            break

    # Merge: remove conflicts across windows (greedy, sort by left endpoint)
    valid_swaps = _validate_no_overlap_intervals(all_swaps)

    # Fallback: if nothing selected, find best single swap across full permutation
    if not valid_swaps:
        best_pair: Optional[Tuple[int, int]] = None
        best_delta = 0.0
        for i in range(n - 1):
            for j in range(i + 1, min(i + w, n)):
                d = compute_endpoint_swap_delta(pi, i, j, Head, Tail, processing_times, base_c)
                if d < best_delta:
                    best_delta = d
                    best_pair = (i, j)
        if best_pair:
            valid_swaps = [best_pair]

    # Apply swaps
    new_pi = pi.copy()
    for i, j in sorted(valid_swaps):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps
