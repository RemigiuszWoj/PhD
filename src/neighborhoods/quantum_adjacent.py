"""Kwantowe sąsiedztwo Adjacent - wybór jednej zamiany przez QUBO.

Formułacja QUBO (one-hot):
    H = Σᵢ δᵢ·xᵢ + P·(Σᵢ xᵢ - 1)²

Macierz QUBO:
    Q[i,i] = δᵢ - P        (koszt + kara za brak wyboru)
    Q[i,j] = 2P            (kara za wybranie więcej niż 1)

Złożoność: n-1 zmiennych, O(n²) współczynników
"""

from typing import Dict, List, Tuple

from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

from src.neighborhoods.common import apply_swaps, compute_deltas
from src.permutation_procesing import c_max


def _solve_qubo(Q: Dict[Tuple[str, str], float], num_reads: int) -> Dict[str, int]:
    """Rozwiązuje QUBO na symulatorze (SimulatedAnnealing).

    W przyszłości można podmienić na DWaveSampler() lub LeapHybridSampler().
    """
    if not Q:
        return {}
    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    result = sampler.sample(bqm, num_reads=num_reads)
    return dict(result.first.sample)


def quantum_adjacent_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 50,
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Kwantowe sąsiedztwo adjacent - wybiera dokładnie jedną zamianę przez QUBO.

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n
        num_reads: Liczba prób dla samplera

    Returns:
        (new_pi, new_cmax, move): Nowa permutacja, Cmax, ruch (i, i+1)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)

    # Buduj QUBO one-hot: dokładnie 1 zamiana
    penalty = 2 * max(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i] - penalty
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty

    # Rozwiąż QUBO
    solution = _solve_qubo(Q, num_reads)
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)

    # Fallback: jeśli solver nie wybrał, wybierz minimum
    idx = selected[0] if selected else min(range(num_vars), key=lambda i: deltas[i])

    new_pi = apply_swaps(pi, [idx])
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, (idx, idx + 1)


# Alias dla kompatybilności wstecznej
def generate_neighbors_adjacent_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 50,
) -> Tuple[List[int], Tuple[int, int]]:
    """Alias dla quantum_adjacent_neighborhood (kompatybilność wsteczna)."""
    new_pi, _, move = quantum_adjacent_neighborhood(pi, processing_times, num_reads)
    return new_pi, move
