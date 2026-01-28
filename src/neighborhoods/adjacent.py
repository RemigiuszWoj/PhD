"""Sąsiedztwo Adjacent - pojedyncze zamiany sąsiednich elementów.

Najprostsze sąsiedztwo: dla permutacji π o n elementach generuje n-1 sąsiadów,
każdy przez zamianę elementów na pozycjach (i, i+1).

Złożoność: O(n) sąsiadów, każdy obliczany w O(m·n) → O(m·n²) total
"""

from typing import Iterator, List, Tuple

from src.neighborhoods.common import swap_jobs
from src.permutation_procesing import c_max


def generate_neighbors_adjacent(
    pi: List[int],
) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generuje wszystkich sąsiadów przez zamiany sąsiednich elementów.

    Generator leniwy - sąsiedzi są generowani na żądanie.

    Args:
        pi: Aktualna permutacja

    Yields:
        (neighbor, move): Sąsiednia permutacja i ruch (i, i+1)
    """
    n = len(pi)
    for i in range(n - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        yield neighbor, (i, i + 1)


def best_adjacent_neighbor(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[int], int, Tuple[int, int]]:
    """Znajduje najlepszego sąsiada w sąsiedztwie adjacent.

    Przegląda wszystkich n-1 sąsiadów i zwraca tego z najmniejszym Cmax.

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n

    Returns:
        (best_pi, best_cmax, move): Najlepsza permutacja, jej Cmax, ruch
    """
    best_pi = None
    best_cmax = float("inf")
    best_move = (-1, -1)

    for neighbor, move in generate_neighbors_adjacent(pi):
        neighbor_cmax = c_max(neighbor, processing_times)
        if neighbor_cmax < best_cmax:
            best_cmax = neighbor_cmax
            best_pi = neighbor
            best_move = move

    if best_pi is None:
        return pi.copy(), c_max(pi, processing_times), (-1, -1)

    return best_pi, best_cmax, best_move
