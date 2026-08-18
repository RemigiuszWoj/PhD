"""Shared QUBO construction for the flow-shop neighborhoods.

One source of truth for candidate enumeration, QUBO assembly and solution
map-back, used by three solver families:
  * ``quantum_qubo``          (D-Wave, single QUBO, improving-only filter),
  * ``quantum_qubo_enhanced`` (D-Wave, windowed, full candidate set),
  * ``gate_qaoa``             (gate-model QAOA).
Each family passes its own flags so behaviour is unchanged; only the solver
that consumes ``Q`` differs.
"""
from .candidates import (
    AdjacentCand,
    IntervalCand,
    enumerate_adjacent_candidates,
    enumerate_interval_candidates,
)
from .assemble import (
    assemble_onehot_qubo,
    assemble_pairwise_qubo,
    assemble_tridiagonal_qubo,
)
from .mapback import selected_indices, selected_intervals, selected_positions

__all__ = [
    "AdjacentCand",
    "IntervalCand",
    "enumerate_adjacent_candidates",
    "enumerate_interval_candidates",
    "assemble_onehot_qubo",
    "assemble_pairwise_qubo",
    "assemble_tridiagonal_qubo",
    "selected_indices",
    "selected_intervals",
    "selected_positions",
]
