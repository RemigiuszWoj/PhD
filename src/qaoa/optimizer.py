"""QAOA parameter optimizer using COBYLA.

Optimizes variational parameters (gammas, betas) by minimizing
expected energy <H_C> estimated from circuit measurements.

Strategy:
    - p=1: analytic warm-start (gamma ~ π/(4*max|J|), beta ~ π/8)
    - p>1: random multi-start
    - Multi-start: run from n_starts initializations, keep best
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import minimize
import cirq

from src.qaoa.circuit import build_qaoa_circuit


def expected_energy(
    counts: Dict[str, int],
    h: List[float],
    J: Dict[Tuple[int, int], float],
) -> float:
    """Compute expected Ising energy <H_C> from measurement counts.

    cirq measures in Z basis: bit 0 → Z=+1, bit 1 → Z=-1.

    Args:
        counts: {bitstring: count} from circuit measurement
        h: Ising linear coefficients
        J: Ising quadratic coefficients

    Returns:
        Expected energy <H_C>
    """
    total = sum(counts.values())
    energy = 0.0

    for bitstring, count in counts.items():
        z = [1 - 2 * int(b) for b in bitstring]
        e = sum(h[i] * z[i] for i in range(len(h)))
        e += sum(jval * z[i] * z[j] for (i, j), jval in J.items())
        energy += e * count / total

    return energy


def optimize_qaoa(
    h: List[float],
    J: Dict[Tuple[int, int], float],
    p: int,
    n_shots: int = 1000,
    n_starts: int = 3,
    simulator: Optional[cirq.Simulator] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float], float]:
    """Optimize QAOA parameters (gammas, betas) via COBYLA.

    Args:
        h: Ising linear coefficients
        J: Ising quadratic coefficients
        p: QAOA depth
        n_shots: Measurement shots per objective evaluation
        n_starts: Number of random restarts
        simulator: cirq.Simulator instance (created if None)
        verbose: Print optimization progress

    Returns:
        (gammas, betas, best_energy)
    """
    if simulator is None:
        simulator = cirq.Simulator()

    def objective(params: np.ndarray) -> float:
        gammas = list(params[:p])
        betas  = list(params[p:])
        circuit = build_qaoa_circuit(h, J, gammas, betas)
        result  = simulator.run(circuit, repetitions=n_shots)
        counts: Dict[str, int] = {}
        for bits in result.measurements["result"]:
            key = "".join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return expected_energy(counts, h, J)

    best_params = None
    best_energy = float("inf")

    for start in range(n_starts):
        if start == 0 and p == 1:
            # Analytic warm-start for p=1
            max_J = max((abs(v) for v in J.values()), default=1.0)
            x0 = np.array([np.pi / (4 * max_J), np.pi / 8])
        else:
            rng = np.random.default_rng(start)
            x0 = rng.uniform(-np.pi, np.pi, 2 * p)

        res = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": 200, "rhobeg": 0.5},
        )

        if verbose:
            print(f"  [optimizer] start={start} energy={res.fun:.4f}")

        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x

    gammas = list(best_params[:p])
    betas  = list(best_params[p:])
    return gammas, betas, best_energy
