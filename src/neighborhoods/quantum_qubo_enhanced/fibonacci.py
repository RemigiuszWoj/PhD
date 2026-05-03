"""Quantum Fibonacci Enhanced neighborhood - full QUBO for large n.

Identical QUBO formulation to quantum_qubo/fibonacci.py (tridiagonal structure,
K = n-1 variables), but with two enhancements for large instances:

1. No delta filter: all swaps included regardless of sign (δ ≥ 0 not removed).
   On real QPU hardware the solver handles the full variable space natively.

2. Supports n up to 499 (K = 498 qubits) — within D-Wave Advantage capacity
   thanks to the tridiagonal (chain) QUBO graph which embeds efficiently.

Feasibility (D-Wave Advantage, ~5627 physical qubits):
    n=20  → K=19,  ~28 physical qubits   ✅
    n=50  → K=49,  ~73 physical qubits   ✅
    n=100 → K=99,  ~148 physical qubits  ✅
    n=200 → K=199, ~298 physical qubits  ✅
    n=500 → K=499, ~748 physical qubits  ✅

Use this class (quantum_qubo_enhanced) for Taillard instances with n ≥ 20.
Use quantum_qubo/fibonacci.py for small instances or classical simulation.
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import (
    apply_swaps,
    compute_deltas,
    solve_qubo,
    validate_no_overlap,
)
from src.permutation_procesing import c_max


def quantum_fibonacci_enhanced(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
) -> Tuple[List[int], int, List[int]]:
    """Quantum Fibonacci Enhanced — full tridiagonal QUBO, no delta filtering.

    Suitable for large n (up to ~500) on real D-Wave QPU.
    For classical simulation (backend='simulator') performance degrades for n > 50;
    use quantum_qubo/fibonacci.py instead for simulation.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of QPU samples (higher = better quality, use ≥ 100 on QPU)
        backend: 'simulator' or 'dwave'
        dwave_token: D-Wave API token (required when backend='dwave')
        solver: D-Wave solver name (None = default Advantage system)
        annealing_time_us: Annealing time in microseconds (None = default 20 µs)
        chain_strength: Chain strength override (None = auto)
        num_spin_reversal_transforms: Spin reversal transforms (None = default)

    Returns:
        (new_pi, new_cmax, swap_positions): New permutation, Cmax,
        list of applied adjacent swap positions
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # Compute deltas for all adjacent swaps — O(m·n)
    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)  # = n - 1

    # Build tridiagonal QUBO — no delta filtering (unlike quantum_qubo version)
    # Q[i,i]   = delta_i          (linear cost)
    # Q[i,i+1] = P                (penalty for overlapping adjacent swaps)
    penalty = sum(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i]
    for i in range(num_vars - 1):
        Q[(f"x{i}", f"x{i + 1}")] = penalty

    # Solve QUBO
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
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)
    valid_swaps = validate_no_overlap(selected)

    # Fallback: if nothing selected, pick best single swap
    if not valid_swaps:
        best_idx = min(range(num_vars), key=lambda i: deltas[i])
        if deltas[best_idx] < 0:
            valid_swaps = [best_idx]

    new_pi = apply_swaps(pi, valid_swaps)
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps
