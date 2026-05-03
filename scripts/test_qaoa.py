"""Smoke test for QAOA Fibonacci pipeline.

Runs end-to-end on a tiny generated instance (n=5, m=3)
to verify imports, circuit building, optimization and sampling work.

Usage:
    .venv311/bin/python3 scripts/test_qaoa.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.taillard_gen import generate_taillard_instance
from src.qaoa.fibonacci import fibonacci_circuit_info
from src.qaoa.runner import run_qaoa_local_search


def test_circuit_info():
    print("=== Circuit resource estimates ===")
    for n in [5, 10, 20, 50, 100]:
        info = fibonacci_circuit_info(n)
        print(f"  n={n:>4}  K={n-1:>3} qubits")
        for p, s in info.items():
            print(f"    p={p}: CZ/layer={s['cz_per_layer']:>4}, "
                  f"total_CZ={s['total_cz']:>4}, "
                  f"error={s['accumulated_error_pct']:>5.2f}%")
    print()


def test_qaoa_small():
    print("=== QAOA local search (n=5, m=3, p=1, noiseless) ===")
    processing_times = generate_taillard_instance(n=5, m=3, seed=42)
    best_pi, best_cmax = run_qaoa_local_search(
        processing_times,
        p=1,
        n_shots=200,
        n_iterations=3,
        use_noise=False,
        verbose=True,
    )
    print(f"  Best permutation: {best_pi}")
    print(f"  Best Cmax: {best_cmax}")
    print()


def test_qaoa_noise():
    print("=== QAOA local search (n=5, m=3, p=1, Willow noise model) ===")
    processing_times = generate_taillard_instance(n=5, m=3, seed=42)
    best_pi, best_cmax = run_qaoa_local_search(
        processing_times,
        p=1,
        n_shots=200,
        n_iterations=2,
        use_noise=True,
        verbose=True,
    )
    print(f"  Best permutation: {best_pi}")
    print(f"  Best Cmax: {best_cmax}")
    print()


if __name__ == "__main__":
    test_circuit_info()
    test_qaoa_small()
    test_qaoa_noise()
    print("All tests passed.")
