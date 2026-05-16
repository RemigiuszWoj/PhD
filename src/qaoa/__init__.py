"""QAOA module for gate-based quantum neighborhood search.

Submodules:
    circuit    - QAOA circuit construction (CZ gates, Willow-native)
    fibonacci  - Fibonacci QUBO → Ising mapping
    optimizer  - COBYLA parameter optimization
    runner     - End-to-end neighborhood evaluation + local search
"""

from src.qaoa.circuit import build_qaoa_circuit, circuit_stats
from src.qaoa.fibonacci import (
    fibonacci_qubo_to_ising,
    fibonacci_qubo_energy,
    fibonacci_circuit_info,
)
from src.qaoa.optimizer import optimize_qaoa, expected_energy
from src.qaoa.runner import qaoa_fibonacci_neighborhood, run_qaoa_local_search

__all__ = [
    "build_qaoa_circuit",
    "circuit_stats",
    "fibonacci_qubo_to_ising",
    "fibonacci_qubo_energy",
    "fibonacci_circuit_info",
    "optimize_qaoa",
    "expected_energy",
    "qaoa_fibonacci_neighborhood",
    "run_qaoa_local_search",
]
