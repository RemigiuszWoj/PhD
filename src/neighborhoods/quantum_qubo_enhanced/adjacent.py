"""Quantum Adjacent Enhanced neighborhood - full QUBO for larger n.

Same one-hot QUBO formulation as quantum_qubo/adjacent.py (K = n-1 variables,
dense all-to-all penalty structure), but without delta filtering and with
higher default num_reads suitable for QPU execution.

Feasibility (D-Wave Advantage, ~5627 physical qubits):
    n=10  → K=9,   ~10 physical qubits   ✅
    n=50  → K=49,  ~300 physical qubits  ✅
    n=100 → K=99,  ~1225 physical qubits ✅
    n=200 → K=199, ~4950 physical qubits ✅ (near limit)
    n=500 → K=499, ~31125 physical qubits ❌ (exceeds D-Wave capacity)

Note: Adjacent has a dense QUBO (complete graph K_{n-1}), which requires
significantly more physical qubits than Fibonacci (chain graph).
For n > 200 use quantum_qubo_enhanced/fibonacci.py instead.
"""

from typing import Dict, List, Tuple

from src.neighborhoods.classical.common import (
    apply_swaps,
    compute_deltas,
    solve_qubo,
)
from src.permutation_procesing import c_max


def quantum_adjacent_enhanced(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Quantum Adjacent Enhanced — one-hot QUBO, no delta filter, large n support.

    Selects exactly one adjacent swap via QUBO minimization.
    Suitable for n up to ~200 on real D-Wave QPU.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        num_reads: Number of QPU samples (use ≥ 100 on QPU)
        backend: 'simulator' or 'dwave'
        dwave_token: D-Wave API token (required when backend='dwave')
        solver: D-Wave solver name (None = default Advantage system)
        annealing_time_us: Annealing time in microseconds
        chain_strength: Chain strength override (None = auto)
        num_spin_reversal_transforms: Spin reversal transforms

    Returns:
        (new_pi, new_cmax, move): New permutation, Cmax, move (i, i+1)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)  # = n - 1

    # One-hot QUBO: exactly one swap selected
    # Q[i,i]   = delta_i - P   (diagonal)
    # Q[i,j]   = 2*P           (off-diagonal, all pairs)
    penalty = 2 * max(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i] - penalty
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty

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

    # Fallback if solver returns empty or multiple
    idx = selected[0] if selected else min(range(num_vars), key=lambda i: deltas[i])

    new_pi = apply_swaps(pi, [idx])
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, (idx, idx + 1)
