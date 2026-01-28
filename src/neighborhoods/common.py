"""Wspólne funkcje dla sąsiedztw Flow Shop.

Ten moduł zawiera podstawowe operacje używane przez wszystkie sąsiedztwa:
- swap_jobs: zamiana dwóch elementów w permutacji
- compute_deltas: obliczanie zmian Cmax dla zamian sąsiednich
- apply_swaps: aplikowanie zamian do permutacji
"""

from typing import List

from src.permutation_procesing import c_max


def swap_jobs(pi: List[int], i: int, j: int) -> List[int]:
    """Zwraca nową permutację z zamienionymi elementami na pozycjach i oraz j.

    Args:
        pi: Permutacja wejściowa
        i: Pierwsza pozycja
        j: Druga pozycja

    Returns:
        Nowa permutacja z zamienionymi elementami
    """
    neighbor = pi.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def compute_deltas(
    pi: List[int],
    processing_times: List[List[int]],
) -> List[float]:
    """Oblicza deltę Cmax dla każdej możliwej zamiany sąsiedniej.

    Delta to różnica między Cmax po zamianie a Cmax przed zamianą.
    Ujemna delta oznacza poprawę (zmniejszenie Cmax).

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n

    Returns:
        Lista delt: δᵢ = Cmax(π po zamianie i,i+1) - Cmax(π)
    """
    base_cmax = c_max(pi, processing_times)
    deltas: List[float] = []

    for i in range(len(pi) - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        neighbor_cmax = c_max(neighbor, processing_times)
        deltas.append(neighbor_cmax - base_cmax)

    return deltas


def apply_swaps(pi: List[int], indices: List[int]) -> List[int]:
    """Stosuje wybrane zamiany sąsiednie do permutacji.

    Każdy indeks i oznacza zamianę elementów na pozycjach i oraz i+1.
    Zamiany są stosowane w kolejności rosnącej indeksów.

    Args:
        pi: Permutacja wejściowa
        indices: Lista pozycji do zamiany (każda zamiana to (i, i+1))

    Returns:
        Nowa permutacja po zamianach
    """
    new_pi = pi.copy()
    for idx in sorted(indices):
        new_pi[idx], new_pi[idx + 1] = new_pi[idx + 1], new_pi[idx]
    return new_pi


def validate_no_overlap(indices: List[int]) -> List[int]:
    """Usuwa nakładające się zamiany (zachowuje tylko nieprzekrywające się).

    Dwie zamiany na pozycjach i oraz i+1 nakładają się, bo dzielą
    pozycję i+1. Ta funkcja zachowuje tylko zamiany które się nie nakładają.

    Args:
        indices: Lista indeksów zamian (posortowana)

    Returns:
        Lista indeksów bez nakładań
    """
    valid: List[int] = []
    last_idx = -2

    for idx in sorted(indices):
        if idx > last_idx + 1:
            valid.append(idx)
            last_idx = idx

    return valid
