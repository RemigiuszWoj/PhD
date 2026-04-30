"""Quantum Motzkin Enhanced neighborhood - windowed QUBO for large n.

Extends quantum_qubo/motzkin.py with a windowed decomposition strategy
analogous to quantum_qubo_enhanced/dynasearch.py, but respecting Motzkin
conflict rules (crossing forbidden, nesting allowed) within each window.

Key difference from Dynasearch Enhanced:
    Motzkin allows nested swaps (i1 < i2 < j2 < j1 is OK),
    while Dynasearch forbids all overlaps including nesting.
    The windowed approach preserves this distinction within each window.
    Cross-window nesting is not captured (limitation of decomposition).

Feasibility: same as dynasearch enhanced — window_size auto-selected
to keep K ≤ 180 variables per window on D-Wave Advantage.
"""

import math
from typing import Dict, List, Optional, Tuple

from src.neighborhoods.classical.common import (
    compute_endpoint_swap_delta,
    compute_head,
    compute_tail,
    solve_qubo,
)
from src.permutation_procesing import c_max

_DWAVE_EFFECTIVE_CAPACITY = 180


def _auto_window_size(n: int, capacity: int = _DWAVE_EFFECTIVE_CAPACITY) -> int:
    w = int((1 + math.sqrt(1 + 8 * capacity)) / 2)
    return max(4, min(w, n))


def _motzkin_conflict(i1: int, j1: int, i2: int, j2: int) -> bool:
    """True if swaps (i1,j1) and (i2,j2) conflict under Motzkin rules.

    Conflicts:
        - Shared endpoint: {i1,j1} ∩ {i2,j2} ≠ ∅
        - Crossing: i1 < i2 < j1 < j2  or  i2 < i1 < j2 < j1

    Allowed (no conflict):
        - Disjoint: j1 < i2  or  j2 < i1
        - Nested:   i1 < i2 < j2 < j1  or  i2 < i1 < j1 < j2
    """
    if {i1, j1} & {i2, j2}:  # shared endpoint
        return True
    # crossing check
    if i1 < i2 < j1 < j2:
        return True
    if i2 < i1 < j2 < j1:
        return True
    return False


def _validate_motzkin(selected: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Greedy feasibility repair: remove conflicting pairs (keep first seen)."""
    valid: List[Tuple[int, int]] = []
    for pair in selected:
        if all(not _motzkin_conflict(pair[0], pair[1], v[0], v[1]) for v in valid):
            valid.append(pair)
    return valid


def _solve_window_motzkin_qubo(
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
    """Build and solve Motzkin QUBO for window [start..end-1]."""
    candidates: List[Tuple[int, int, float]] = []
    for i in range(start, end - 1):
        for j in range(i + 1, end):
            delta = compute_endpoint_swap_delta(
                pi, i, j, Head, Tail, processing_times, base_c
            )
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
            if _motzkin_conflict(i1, j1, i2, j2):
                Q[(f"x{k}", f"x{l}")] = penalty
            # nested/disjoint pairs → Q[k,l] = 0 (no penalty)

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
    raw = [(candidates[k][0], candidates[k][1]) for k in selected_k]
    return _validate_motzkin(raw)


def quantum_motzkin_enhanced(
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
    """Quantum Motzkin Enhanced — windowed QUBO decomposition for large n.

    For n ≤ window_size: single QUBO over full permutation (same as
    quantum_qubo/motzkin.py but without delta filtering).
    For large n: overlapping windows, Motzkin conflict rules within each window,
    greedy cross-window conflict resolution.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        window_size: Window size. None = auto (~19 for D-Wave Advantage).
        overlap_ratio: Window overlap fraction (0.5 = 50% overlap).
        num_reads: QPU samples per window
        backend: 'simulator' or 'dwave'
        dwave_token: D-Wave API token
        solver: D-Wave solver name
        annealing_time_us: Annealing time in µs
        chain_strength: Chain strength override
        num_spin_reversal_transforms: Spin reversal transforms

    Returns:
        (new_pi, new_cmax, applied_swaps): New permutation, Cmax,
        list of (i, j) endpoint swap pairs applied
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    w = window_size if window_size is not None else _auto_window_size(n)
    w = min(w, n)

    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    m_machines = len(processing_times)
    base_c = Head[m_machines - 1][n - 1]

    all_swaps: List[Tuple[int, int]] = []
    step = max(1, int(w * (1 - overlap_ratio)))

    for start in range(0, n, step):
        end = min(start + w, n)
        if end - start < 2:
            break
        window_swaps = _solve_window_motzkin_qubo(
            pi, processing_times, start, end,
            Head, Tail, base_c,
            num_reads, backend, dwave_token, solver,
            annealing_time_us, chain_strength, num_spin_reversal_transforms,
        )
        all_swaps.extend(window_swaps)
        if end == n:
            break

    # Global conflict resolution (greedy, respecting Motzkin rules)
    valid_swaps = _validate_motzkin(all_swaps)

    # Fallback: best single swap
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

    new_pi = pi.copy()
    for i, j in sorted(valid_swaps, key=lambda x: x[0]):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps
