"""Gate-model (QAOA) neighborhood family for the permutation flow shop.

The gate-model analog of the ``quantum_qubo`` / ``quantum_qubo_enhanced``
families. It reuses the shared QUBO construction from ``common_qubo`` (so the
matrix Q is identical to the annealer's) and solves it with a fixed-angle
QAOA circuit instead of a quantum annealer. Layout mirrors how the annealing
solver lives inside ``neighborhoods`` (``common.solve_qubo``): the QAOA engine
(``circuit``, ``angles``, ``solve``) sits next to the four neighborhood files.
"""
from .circuit import (
    bitstring_to_assignment,
    build_qaoa_circuit,
    ising_hamiltonian,
    normalize_ising,
    qubo_energy,
    qubo_to_ising,
)
from .angles import interp_seed, optimize_neighborhood
from .solve import solve_qaoa

__all__ = [
    "qubo_to_ising",
    "normalize_ising",
    "ising_hamiltonian",
    "build_qaoa_circuit",
    "bitstring_to_assignment",
    "qubo_energy",
    "optimize_neighborhood",
    "interp_seed",
    "solve_qaoa",
]
