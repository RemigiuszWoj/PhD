"""QAOA runner for PFSP Fibonacci neighborhood.

End-to-end pipeline:
    permutation → QUBO → Ising → QAOA circuit → optimize → sample → new permutation

Backends:
    - cirq.Simulator          (noiseless, local)
    - cirq.DensityMatrixSimulator + noise model  (Willow noise approximation)
    - TODO: cirq_google.engine (real Willow QPU, requires access token)
"""

from typing import List, Tuple, Optional
import numpy as np
import cirq

from src.qaoa.fibonacci import fibonacci_qubo_to_ising, fibonacci_qubo_energy
from src.qaoa.circuit import build_qaoa_circuit, circuit_stats
from src.qaoa.optimizer import optimize_qaoa
from src.neighborhoods.common import apply_swaps, validate_no_overlap
from src.permutation_procesing import c_max


# Willow Device 2 noise parameters
WILLOW_CZ_ERROR      = 0.0014   # two-qubit gate depolarizing error
WILLOW_SQ_ERROR      = 0.0003   # single-qubit gate error (unused in simple model)
WILLOW_READOUT_ERROR = 0.006    # readout error per qubit (unused in simple model)


def make_willow_noise_model() -> cirq.ConstantQubitNoiseModel:
    """Approximate Willow Device 2 noise model (depolarizing on all gates).

    T1/T2 decay not modeled — circuit times << T1 = 98 µs.
    """
    return cirq.ConstantQubitNoiseModel(
        qubit_noise_gate=cirq.depolarize(WILLOW_CZ_ERROR)
    )


def qaoa_fibonacci_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
    p: int = 1,
    n_shots: int = 1000,
    n_starts: int = 3,
    use_noise: bool = False,
    verbose: bool = False,
) -> Tuple[List[int], int, List[int]]:
    """One QAOA Fibonacci neighborhood evaluation.

    Args:
        pi: Current permutation
        processing_times: m × n processing times matrix
        p: QAOA depth
        n_shots: Measurement shots per circuit evaluation
        n_starts: COBYLA multi-start restarts
        use_noise: Apply Willow noise model
        verbose: Print progress

    Returns:
        (new_pi, new_cmax, applied_swaps)
    """
    n = len(pi)
    if n < 2:
        return pi.copy(), c_max(pi, processing_times), []

    # Step 1: QUBO → Ising
    deltas, h, J = fibonacci_qubo_to_ising(pi, processing_times)
    P = sum(abs(d) for d in deltas) + 1.0

    if verbose:
        stats = circuit_stats(h, J, p)
        print(f"[QAOA Fibonacci] n={n}, K={len(h)}, p={p}")
        print(f"  CZ/layer={stats['cz_per_layer']}, "
              f"total_CZ={stats['total_cz']}, "
              f"error={stats['accumulated_error_pct']}%")

    # Step 2: Simulator
    if use_noise:
        simulator = cirq.DensityMatrixSimulator(noise=make_willow_noise_model())
    else:
        simulator = cirq.Simulator()

    # Step 3: Optimize parameters
    gammas, betas, best_energy = optimize_qaoa(
        h, J, p,
        n_shots=n_shots,
        n_starts=n_starts,
        simulator=simulator,
        verbose=verbose,
    )

    if verbose:
        print(f"  Optimized energy: {best_energy:.4f}")

    # Step 4: Sample from optimized circuit
    circuit = build_qaoa_circuit(h, J, gammas, betas)
    result  = simulator.run(circuit, repetitions=n_shots)

    # Step 5: Best bitstring by QUBO energy
    best_bits = None
    best_qubo_energy = float("inf")
    for bits in result.measurements["result"]:
        x = list(map(int, bits))
        e = fibonacci_qubo_energy(x, deltas, P)
        if e < best_qubo_energy:
            best_qubo_energy = e
            best_bits = x

    if best_bits is None:
        return pi.copy(), c_max(pi, processing_times), []

    # Step 6: Extract valid non-overlapping swaps
    selected   = [i for i, b in enumerate(best_bits) if b == 1]
    valid_swaps = validate_no_overlap(selected)

    # Fallback: best single improving swap
    if not valid_swaps:
        best_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        if deltas[best_idx] < 0:
            valid_swaps = [best_idx]

    new_pi    = apply_swaps(pi, valid_swaps)
    new_cmax  = c_max(new_pi, processing_times)

    if verbose:
        print(f"  Swaps: {valid_swaps}, "
              f"Cmax: {c_max(pi, processing_times)} → {new_cmax}")

    return new_pi, new_cmax, valid_swaps


def run_qaoa_local_search(
    processing_times: List[List[int]],
    p: int = 1,
    n_shots: int = 1000,
    n_iterations: int = 10,
    use_noise: bool = False,
    verbose: bool = True,
) -> Tuple[List[int], int]:
    """Greedy QAOA local search: repeatedly apply Fibonacci neighborhood.

    Simple end-to-end test — not a full metaheuristic.

    Args:
        processing_times: m × n processing times matrix
        p: QAOA depth
        n_shots: Shots per circuit
        n_iterations: Number of neighborhood evaluations
        use_noise: Apply Willow noise model
        verbose: Print progress

    Returns:
        (best_pi, best_cmax)
    """
    n = len(processing_times[0])
    pi       = list(range(n))
    best_pi  = pi.copy()
    best_cmax = c_max(pi, processing_times)

    if verbose:
        print(f"[run_qaoa_local_search] n={n}, p={p}, "
              f"noise={use_noise}, iterations={n_iterations}")
        print(f"  Initial Cmax: {best_cmax}")

    for it in range(n_iterations):
        new_pi, new_cmax, swaps = qaoa_fibonacci_neighborhood(
            pi, processing_times,
            p=p, n_shots=n_shots,
            use_noise=use_noise,
            verbose=verbose,
        )
        if new_cmax < best_cmax:
            best_cmax = new_cmax
            best_pi   = new_pi.copy()
            if verbose:
                print(f"  iter={it}: improved → {best_cmax}")
        pi = new_pi

    if verbose:
        print(f"  Final best Cmax: {best_cmax}")

    return best_pi, best_cmax
