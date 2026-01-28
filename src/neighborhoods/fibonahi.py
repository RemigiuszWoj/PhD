"""Sąsiedztwo Fibonahi - nieprzekrywające się zamiany sąsiednie.

Wybiera optymalny zbiór nieprzekrywających się zamian sąsiednich
minimalizujący sumaryczną zmianę Cmax.

Nazwa pochodzi od ciągu Fibonacciego: liczba możliwych zbiorów
nieprzekrywających się zamian z n-1 pozycji wynosi F_{n+1}.

Metoda: Programowanie dynamiczne O(n)
Złożoność: O(m·n) dla delt + O(n) dla DP = O(m·n) total
"""

from typing import Dict, List, Tuple

from src.neighborhoods.common import apply_swaps
from src.permutation_procesing import c_max


def fibonahi_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[int], int]:
    """Znajduje najlepszy zbiór nieprzekrywających się zamian metodą DP.

    Algorytm DP:
        solve(pos) = min(
            solve(pos + 1),                    # pomiń zamianę na pos
            δ[pos] + solve(pos + 2)            # weź zamianę, przeskocz następną
        )

    UWAGA: Delty są obliczane niezależnie dla oryginalnej permutacji.
    To jest przybliżenie - w rzeczywistości po wykonaniu jednej zamiany
    delty innych zamian mogą się zmienić.

    Args:
        pi: Aktualna permutacja
        processing_times: Macierz czasów m × n

    Returns:
        (new_pi, new_cmax): Nowa permutacja i jej Cmax
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times)

    base_c = c_max(pi, processing_times)

    # Oblicz delty
    candidates: List[Tuple[int, float]] = []  # (pozycja, delta)
    for i in range(n - 1):
        neighbor = pi.copy()
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        delta = c_max(neighbor, processing_times) - base_c
        candidates.append((i, delta))

    if not candidates:
        return pi.copy(), base_c

    m = len(candidates)
    memo: Dict[int, Tuple[float, Tuple[int, ...]]] = {}

    def solve(pos: int) -> Tuple[float, Tuple[int, ...]]:
        """Rekurencyjne DP z memoizacją."""
        if pos >= m:
            return 0.0, ()
        if pos in memo:
            return memo[pos]

        idx, delta = candidates[pos]

        # Opcja 1: pomiń ten ruch
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set

        # Opcja 2: weź ten ruch → przeskocz następny (nakładający się)
        next_pos = pos + 1
        while next_pos < m and candidates[next_pos][0] == idx + 1:
            next_pos += 1

        take_rest_val, take_rest_set = solve(next_pos)
        take_val = delta + take_rest_val

        if take_val < best_val:
            best_val = take_val
            best_set = (idx,) + take_rest_set

        memo[pos] = (best_val, best_set)
        return memo[pos]

    total_delta, chosen = solve(0)

    # Fallback: jeśli nic nie wybrano, wybierz najlepszą pojedynczą zamianę
    if not chosen:
        best_idx = min(range(len(candidates)), key=lambda i: candidates[i][1])
        chosen = (best_idx,)

    # Zastosuj wybrane zamiany
    new_pi = apply_swaps(pi, list(chosen))
    final_c = c_max(new_pi, processing_times)

    return new_pi, final_c
