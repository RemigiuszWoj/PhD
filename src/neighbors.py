from typing import Iterator, List, Tuple


def swap_jobs(pi: List[int], i: int, j: int) -> List[int]:
    """Return a new permutation with jobs at positions i and j swapped."""
    neighbor = pi.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def generate_neighbors_swap(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = swap_jobs(pi, i, i + 1)
            yield neighbor, (i, j)


def generate_neighbors_adjacent(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        yield neighbor, (i, i + 1)
