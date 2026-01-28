"""Moduł sąsiedztw dla problemu Flow Shop.

Struktura:
- common.py: wspólne funkcje (swap_jobs, compute_deltas, apply_swaps)
- adjacent.py: sąsiedztwo adjacent (klasyczne)
- fibonahi.py: sąsiedztwo fibonahi (klasyczne, DP)
- dynasearch.py: sąsiedztwo dynasearch
- motzkin.py: sąsiedztwo motzkin
- quantum_adjacent.py: kwantowe adjacent (QUBO ONE_HOT)
- quantum_fibonahi.py: kwantowe fibonahi (QUBO NO_OVERLAP)
- boundaries.py: compute_prefix_boundaries
"""

# Klasyczne sąsiedztwa
from src.neighborhoods.adjacent import (
    best_adjacent_neighbor,
    generate_neighbors_adjacent,
)

# Boundaries
from src.neighborhoods.boundaries import compute_prefix_boundaries

# Wspólne funkcje
from src.neighborhoods.common import (
    apply_swaps,
    compute_deltas,
    swap_jobs,
    validate_no_overlap,
)
from src.neighborhoods.dynasearch import dynasearch_full
from src.neighborhoods.fibonahi import fibonahi_neighborhood
from src.neighborhoods.motzkin import motzkin_neighborhood_full

# Kwantowe sąsiedztwa
from src.neighborhoods.quantum_adjacent import (
    generate_neighbors_adjacent_qubo,
    quantum_adjacent_neighborhood,
)
from src.neighborhoods.quantum_fibonahi import (
    generate_neighbors_fibonahi_qubo,
    quantum_fibonahi_neighborhood,
)

__all__ = [
    # Common
    "swap_jobs",
    "compute_deltas",
    "apply_swaps",
    "validate_no_overlap",
    # Classic neighborhoods
    "generate_neighbors_adjacent",
    "best_adjacent_neighbor",
    "fibonahi_neighborhood",
    "dynasearch_full",
    "motzkin_neighborhood_full",
    # Quantum neighborhoods
    "quantum_adjacent_neighborhood",
    "quantum_fibonahi_neighborhood",
    "generate_neighbors_adjacent_qubo",
    "generate_neighbors_fibonahi_qubo",
    # Boundaries
    "compute_prefix_boundaries",
]
