from typing import Iterator, List, Tuple


def generate_neighbors_swap(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = pi.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor, (i, j)


# def generate_neighbors_swap(pi: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
#     """Generate neighbors by swapping any two jobs in pi."""
#     neighbors = []
#     n = len(pi)
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             neighbor = pi.copy()
#             neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#             neighbors.append((neighbor, (i, j)))
#     return neighbors


def generate_neighbors_adjacent(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        neighbor = pi.copy()
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        yield neighbor, (i, i + 1)


# def generate_neighbors_adjacent(pi: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
#     """Generate neighbors by swapping only adjacent jobs in pi."""
#     neighbors = []
#     n = len(pi)
#     for i in range(n - 1):
#         neighbor = pi.copy()
#         neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
#         neighbors.append((neighbor, (i, i + 1)))
#     return neighbors
