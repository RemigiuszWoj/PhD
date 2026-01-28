from typing import Dict, List, Tuple

from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler

from src.neighbors import swap_jobs
from src.permutation_procesing import c_max


def build_adjacent_qubo(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[Dict[Tuple[str, str], float], List[float]]:
    """
    Buduje macierz QUBO dla wyboru najlepszej zamiany sąsiedniej.

    Problem: Wybierz dokładnie jedną zamianę (i, i+1) minimalizującą Cmax.

    Formułacja QUBO:
        min  Σᵢ δᵢ·xᵢ + P·(Σᵢ xᵢ - 1)²

    gdzie:
        - xᵢ ∈ {0,1} - czy wykonać zamianę na pozycji i
        - δᵢ = Cmax(π po zamianie i) - Cmax(π)
        - P = kara za naruszenie ograniczenia one-hot

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n

    Returns:
        (Q, deltas): Macierz QUBO, lista delt dla każdej zamiany
    """
    n = len(pi)
    base_cmax = c_max(pi, processing_times)

    # Oblicz deltę dla każdej zamiany sąsiedniej
    deltas = []
    for i in range(n - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        neighbor_cmax = c_max(neighbor, processing_times)
        delta = neighbor_cmax - base_cmax
        deltas.append(delta)

    # Dobór kary - większa niż max różnica
    max_abs_delta = max(abs(d) for d in deltas) if deltas else 1
    penalty = 2 * max_abs_delta + 1

    # Budowa macierzy QUBO
    Q: Dict[Tuple[str, str], float] = {}
    num_vars = n - 1

    # Wyrazy liniowe (diagonala): (δᵢ - P)·xᵢ
    for i in range(num_vars):
        Q[(f"x{i}", f"x{i}")] = deltas[i] - penalty

    # Wyrazy kwadratowe (poza diagonalą): 2P·xᵢ·xⱼ
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(f"x{i}", f"x{j}")] = 2 * penalty

    return Q, deltas


def solve_qubo_simulator(
    Q: Dict[Tuple[str, str], float],
    num_reads: int = 50,
) -> Dict[str, int]:
    """
    Rozwiązuje problem QUBO na symulatorze (Simulated Annealing).

    W przyszłości ta funkcja zostanie zastąpiona przez:
    - solve_qubo_qpu() - prawdziwy komputer kwantowy D-Wave

    Args:
        Q: Macierz QUBO
        num_reads: Liczba prób dla samplera

    Returns:
        solution: Słownik {nazwa_zmiennej: wartość}
    """

    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()
    result = sampler.sample(bqm, num_reads=num_reads)

    return dict(result.first.sample)


def generate_neighbors_adjacent_qubo(
    pi: List[int],
    processing_times: List[List[int]],
    num_reads: int = 50,
) -> Tuple[List[int], Tuple[int, int]]:
    """
    Kwantowa wersja generate_neighbors_adjacent.

    Zamiast zwracać wszystkich sąsiadów, od razu zwraca najlepszego
    wybranego przez QUBO solver.

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n
        num_reads: Liczba prób dla samplera

    Returns:
        (neighbor, move): Najlepsza permutacja i ruch (i, i+1)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), (-1, -1)

    # 1. Zbuduj QUBO
    Q, deltas = build_adjacent_qubo(pi, processing_times)

    # 2. Rozwiąż QUBO (na symulatorze, później na QPU)
    solution = solve_qubo_simulator(Q, num_reads=num_reads)

    # 3. Znajdź wybraną zamianę
    selected_idx = None
    for var_name, value in solution.items():
        if value == 1:
            selected_idx = int(var_name[1:])  # "x3" -> 3
            break

    # Fallback: jeśli solver nie wybrał nic, wybierz minimum klasycznie
    if selected_idx is None:
        selected_idx = deltas.index(min(deltas))

    # 4. Zwróć wynik
    neighbor = swap_jobs(pi, selected_idx, selected_idx + 1)
    move = (selected_idx, selected_idx + 1)

    return neighbor, move
