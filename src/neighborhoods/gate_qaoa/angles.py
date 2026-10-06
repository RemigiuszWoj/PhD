"""Offline fixed-angle calibration for QAOA neighborhood evaluation.

Given a set of representative (small, windowed) QUBOs for one neighborhood,
find a single angle schedule (gamma_1..gamma_p, beta_1..beta_p) per depth p
that minimizes the mean normalized cost expectation <H_C> over the set. The
angles are then frozen and reused for every move of that neighborhood
(see solve.py). No brute force: <H_C> is read exactly from the statevector,
so calibration scales to any window that fits the simulator.

Depth strategy:
  * p = 1: dense 2-D grid search (only two angles) -- robust, no seeds.
  * p >= 2: continuous optimization (Nelder-Mead) seeded by INTERP -- the
    optimal p-1 schedule, linearly resampled to length p (Zhou et al. 2020),
    which keeps higher-depth optimization cheap and stable.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .circuit import (
    QUBO,
    build_qaoa_circuit,
    ising_hamiltonian,
    normalize_ising,
    qubo_to_ising,
)


def _prepare(training: Sequence[QUBO]):
    """Pre-build the normalized Ising data for each training QUBO once.

    Returns a list of (K, h_norm, J_norm, H_op, l1) tuples. Dividing the
    expectation by the L1 norm ``l1 = sum|h| + sum|J|`` keeps every QUBO's
    contribution in roughly [-1, 0] regardless of size, so no single window
    dominates the mean.
    """
    prepared = []
    for Q in training:
        variables, h, J, _ = qubo_to_ising(Q)
        K = len(variables)
        if K == 0:
            continue
        hn, Jn, _ = normalize_ising(h, J)
        l1 = float(np.sum(np.abs(hn)) + sum(abs(v) for v in Jn.values())) or 1.0
        prepared.append((K, hn, Jn, ising_hamiltonian(K, hn, Jn), l1))
    return prepared


def _objective(prepared, gammas: Sequence[float], betas: Sequence[float]) -> float:
    """Mean normalized cost expectation over the training QUBOs (to minimize)."""
    total = 0.0
    for K, hn, Jn, H_op, l1 in prepared:
        qc = build_qaoa_circuit(K, hn, Jn, gammas, betas)
        exp = Statevector(qc).expectation_value(H_op).real
        total += exp / l1
    return total / max(len(prepared), 1)


def interp_seed(vec: Sequence[float]) -> List[float]:
    """INTERP: resample a length-p angle vector to length p+1 by interpolation."""
    p = len(vec)
    if p == 0:
        return [0.0]
    return list(np.interp(np.linspace(0.0, 1.0, p + 1),
                          np.linspace(0.0, 1.0, p), np.asarray(vec, float)))


def _grid_p1(prepared, n_gamma: int = 25, n_beta: int = 13):
    """Exhaustive 2-D grid for p=1. gamma in [0, pi], beta in [0, pi/2]."""
    best = (None, 0.0, 0.0)
    for g in np.linspace(0.0, np.pi, n_gamma):
        for b in np.linspace(0.0, np.pi / 2, n_beta):
            val = _objective(prepared, [g], [b])
            if best[0] is None or val < best[0]:
                best = (val, g, b)
    return best[0], [best[1]], [best[2]]


def _run_nm(prepared, seed: np.ndarray, p: int, maxiter: int):
    def fun(x):
        return _objective(prepared, x[:p], x[p:])
    res = minimize(fun, seed, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
    return float(res.fun), list(res.x[:p]), list(res.x[p:])


def _optimize_p(prepared, prev_g: Sequence[float], prev_b: Sequence[float],
                maxiter: int, rng: np.random.Generator, n_restarts: int = 2):
    """Depth p = len(prev)+1 via multi-start Nelder-Mead, keep the best.

    Seeds: (1) the previous schedule with a zero last layer, which reproduces
    the depth p-1 state exactly and therefore guarantees the depth-p optimum is
    no worse than depth p-1; (2) the INTERP resample; (3..) INTERP with small
    jitter to escape local minima.
    """
    p = len(prev_g) + 1
    seeds: List[np.ndarray] = [
        np.concatenate([list(prev_g) + [0.0], list(prev_b) + [0.0]]),  # monotonicity guard
        np.concatenate([interp_seed(prev_g), interp_seed(prev_b)]),    # INTERP
    ]
    base = seeds[1]
    for _ in range(n_restarts):
        seeds.append(base + rng.normal(0.0, 0.15, size=base.shape))
    best = None
    for s in seeds:
        obj, g, b = _run_nm(prepared, s, p, maxiter)
        if best is None or obj < best[0]:
            best = (obj, g, b)
    return best


def optimize_neighborhood(training: Sequence[QUBO], p_max: int = 5,
                          n_gamma: int = 25, n_beta: int = 13,
                          maxiter: int = 300, seed: int = 0) -> Dict[int, Dict]:
    """Calibrate fixed angles for p = 1..p_max on one neighborhood's training set.

    Returns ``{p: {"gamma": [...], "beta": [...], "objective": float}}``. p=1 is
    a grid search; each deeper p is a multi-start Nelder-Mead seeded from the
    depth p-1 result, so the objective is non-increasing in p by construction.
    """
    prepared = _prepare(training)
    rng = np.random.default_rng(seed)
    out: Dict[int, Dict] = {}
    prev_g: List[float] = []
    prev_b: List[float] = []
    for p in range(1, p_max + 1):
        if p == 1:
            obj, g, b = _grid_p1(prepared, n_gamma, n_beta)
        else:
            obj, g, b = _optimize_p(prepared, prev_g, prev_b, maxiter, rng)
        out[p] = {"gamma": [float(x) for x in g],
                  "beta": [float(x) for x in b],
                  "objective": float(obj)}
        prev_g, prev_b = g, b
    return out
