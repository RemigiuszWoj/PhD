"""Kwantowe sąsiedztwo Fibonahi - wybór nieprzekrywających się zamian przez QUBO.

Formułacja QUBO (no-overlap):
    H = Σᵢ δᵢ·xᵢ + P·Σᵢ xᵢ·xᵢ₊₁

Macierz QUBO:
    Q[i,i] = δᵢ            (tylko koszt)
    Q[i,i+1] = P           (kara za nakładanie)

Złożoność: n-1 zmiennych, O(n) współczynników (graf łańcuchowy)
Liczba poprawnych zbiorów zamian = F_{n+1} (ciąg Fibonacciego)
"""

from typing import Dict, List, Tuple

from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

from src.neighborhoods.common import apply_swaps, compute_deltas, validate_no_overlap
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


def quantum_fibonahi_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
) -> Tuple[List[int], int, List[int]]:
    """Kwantowe sąsiedztwo fibonahi - wybiera nieprzekrywające się zamiany przez QUBO.

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n
        num_reads: Liczba prób dla samplera

    Returns:
        (new_pi, new_cmax, swaps): Nowa permutacja, Cmax, lista pozycji zamian
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    deltas = compute_deltas(pi, processing_times)
    num_vars = len(deltas)

    # Buduj QUBO no-overlap: nieprzekrywające się zamiany
    penalty = sum(abs(d) for d in deltas) + 1
    Q: Dict[Tuple[str, str], float] = {}
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i]
    for i in range(num_vars - 1):
        Q[(f"x{i}", f"x{i + 1}")] = penalty

    # Rozwiąż QUBO
    solution = _solve_qubo(Q, num_reads)
    selected = sorted(int(v[1:]) for v, val in solution.items() if val == 1)
    valid_swaps = validate_no_overlap(selected)

    # Fallback: jeśli nic nie wybrano, wybierz najlepszą pojedynczą
    if not valid_swaps:
        best_idx = min(range(num_vars), key=lambda i: deltas[i])
        if deltas[best_idx] < 0:
            valid_swaps = [best_idx]

    new_pi = apply_swaps(pi, valid_swaps)
    new_cmax = c_max(new_pi, processing_times)
    return new_pi, new_cmax, valid_swaps


# Alias dla kompatybilności wstecznej
def generate_neighbors_fibonahi_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 100,
) -> Tuple[List[int], int, List[int]]:
    """Alias dla quantum_fibonahi_neighborhood (kompatybilność wsteczna)."""
    return quantum_fibonahi_neighborhood(pi, processing_times, num_reads)
