"""Solution -> selection helpers shared by every family.

These turn a solver result (``{"x0": 1, "x1": 0, ...}``) into the objects a
neighbourhood needs. Neighbourhood-specific validation and fallbacks stay
in the neighbourhood module; only the raw extraction is shared.
"""
from typing import Dict, List, Tuple


def selected_indices(solution: Dict[str, int]) -> List[int]:
    """Sorted local variable indices set to 1 (e.g. {"x2":1,"x0":1} -> [0, 2])."""
    return sorted(int(v[1:]) for v, val in solution.items() if val == 1)


def selected_intervals(
    solution: Dict[str, int],
    candidates: List[Tuple[int, int, float]],
) -> List[Tuple[int, int]]:
    """Endpoint-swap pairs (i, j) chosen by the solution (Dynasearch / Motzkin)."""
    return [(candidates[k][0], candidates[k][1]) for k in selected_indices(solution)]


def selected_positions(
    solution: Dict[str, int],
    positions: List[int],
) -> List[int]:
    """Original swap positions chosen by the solution (Adjacent / Fibonacci)."""
    return [positions[k] for k in selected_indices(solution)]
