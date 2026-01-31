"""Flow Shop neighborhoods module.

Structure:
- common.py: shared functions (swap_jobs, compute_head, compute_tail, compute_deltas, apply_swaps)
- adjacent.py: adjacent neighborhood (classic)
- fibonahi.py: fibonahi neighborhood (classic, DP)
- dynasearch.py: dynasearch neighborhood
- motzkin.py: motzkin neighborhood
- quantum_adjacent.py: quantum adjacent (QUBO ONE_HOT)
- quantum_fibonahi.py: quantum fibonahi (QUBO NO_OVERLAP)
- boundaries.py: compute_prefix_boundaries
"""

# Classic neighborhoods
from src.neighborhoods.adjacent import (
    best_adjacent_neighbor,
    generate_neighbors_adjacent,
)

# Boundaries
from src.neighborhoods.boundaries import compute_prefix_boundaries

# Shared functions
from src.neighborhoods.common import (
    apply_swaps,
    compute_deltas,
    compute_head,
    compute_head_and_tail,
    compute_tail,
    solve_qubo,
    swap_jobs,
    validate_no_overlap,
)
from src.neighborhoods.dynasearch import dynasearch_full
from src.neighborhoods.fibonahi import fibonahi_neighborhood_topk
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
    "compute_head",
    "compute_tail",
    "compute_head_and_tail",
    "compute_deltas",
    "apply_swaps",
    "validate_no_overlap",
    "solve_qubo",
    # Classic neighborhoods
    "generate_neighbors_adjacent",
    "best_adjacent_neighbor",
    "fibonahi_neighborhood_topk",
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
