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
from .adjacent import gate_adjacent_neighborhood
from .dynasearch import gate_dynasearch_neighborhood
from .fibonacci import gate_fibonacci_neighborhood
from .motzkin import gate_motzkin_neighborhood

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
    "gate_adjacent_neighborhood",
    "gate_dynasearch_neighborhood",
    "gate_fibonacci_neighborhood",
    "gate_motzkin_neighborhood",
]
