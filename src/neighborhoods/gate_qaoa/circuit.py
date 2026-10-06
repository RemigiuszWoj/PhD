"""QAOA circuit construction for QUBO neighborhood evaluation.

QUBO -> Ising -> depth-p QAOA circuit. The same functions serve the noiseless
offline angle calibration (angles.py) and real hardware (the circuit
transpiles to IBM Heron); only the executing backend differs (see solve.py). The variable convention x = (1 - Z)/2 makes a measured
computational-basis bit value equal to the QUBO variable value, so a sampled
bitstring maps straight back to a swap selection.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

QUBO = Dict[Tuple[str, str], float]


def qubo_to_ising(Q: QUBO, variables: Sequence[str] | None = None):
    """Map a QUBO ``E(x) = x^T Q x`` (x in {0,1}) to an Ising Hamiltonian.

    Substitutes ``x_i = (1 - Z_i) / 2``. Returns
    ``(variables, h, J, const)`` where ``h`` is a length-K array of linear
    Z-fields, ``J`` maps ``(i, j)`` with ``i < j`` to a ZZ coupling, and
    ``const`` is the identity offset (irrelevant to the optimum).
    """
    if variables is None:
        variables = sorted({v for pair in Q for v in pair})
    variables = list(variables)
    index = {v: k for k, v in enumerate(variables)}
    K = len(variables)

    h = np.zeros(K)
    J: Dict[Tuple[int, int], float] = {}
    const = 0.0
    for (a, b), w in Q.items():
        ia, ib = index[a], index[b]
        if ia == ib:                       # linear term  w * x_i
            const += w / 2.0
            h[ia] += -w / 2.0
        else:                              # quadratic term  w * x_i x_j
            if ia > ib:
                ia, ib = ib, ia
            const += w / 4.0
            h[ia] += -w / 4.0
            h[ib] += -w / 4.0
            J[(ia, ib)] = J.get((ia, ib), 0.0) + w / 4.0
    return variables, h, J, const


def normalize_ising(h: np.ndarray, J: Dict[Tuple[int, int], float]):
    """Rescale h, J to unit maximum |coefficient|.

    Fixed QAOA angles only transfer between instances when the cost
    Hamiltonian has a common scale; the penalty weight P differs from move to
    move, so we divide it out. Returns ``(h_norm, J_norm, scale)``.
    """
    scale = max(
        float(np.max(np.abs(h))) if h.size else 0.0,
        max((abs(v) for v in J.values()), default=0.0),
        1e-12,
    )
    return h / scale, {k: v / scale for k, v in J.items()}, scale


def ising_hamiltonian(K: int, h: np.ndarray, J: Dict[Tuple[int, int], float]) -> SparsePauliOp:
    """Cost Hamiltonian ``H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j``."""
    terms: List[Tuple[str, float]] = []
    for i in range(K):
        if abs(h[i]) > 1e-12:
            label = ["I"] * K
            label[K - 1 - i] = "Z"
            terms.append(("".join(label), float(h[i])))
    for (i, j), v in J.items():
        if abs(v) > 1e-12:
            label = ["I"] * K
            label[K - 1 - i] = "Z"
            label[K - 1 - j] = "Z"
            terms.append(("".join(label), float(v)))
    if not terms:
        terms.append(("I" * K, 0.0))
    return SparsePauliOp.from_list(terms)


def build_qaoa_circuit(
    K: int,
    h: np.ndarray,
    J: Dict[Tuple[int, int], float],
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    measure: bool = False,
) -> QuantumCircuit:
    """Depth-p QAOA circuit, ``p = len(gammas) = len(betas)``.

    Hadamard init, then for each layer l a cost evolution ``e^{-i gamma_l H_C}``
    (``RZ(2 gamma h_i)`` + ``RZZ(2 gamma J_ij)``) and a mixer
    ``e^{-i beta_l H_M}`` (``RX(2 beta)``). Native gates only, so it transpiles
    directly to IBM Heron. ``measure=True`` adds a final measurement (hardware
    / shot-based sampling); leave it False for offline angle calibration.
    """
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have equal length (= p)")
    qc = QuantumCircuit(K)
    qc.h(range(K))
    for gamma, beta in zip(gammas, betas):
        for i in range(K):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * gamma * float(h[i]), i)
        for (i, j), v in J.items():
            if abs(v) > 1e-12:
                qc.rzz(2.0 * gamma * float(v), i, j)
        for i in range(K):
            qc.rx(2.0 * beta, i)
    if measure:
        qc.measure_all()
    return qc


def bitstring_to_assignment(bitstring: str, variables: Sequence[str]) -> Dict[str, int]:
    """Qiskit little-endian bitstring -> ``{var: 0/1}``.

    Qubit k is character ``bitstring[K-1-k]``; with x = (1 - Z)/2 the measured
    bit equals the QUBO variable, so no sign flip is needed.
    """
    K = len(variables)
    return {variables[k]: int(bitstring[K - 1 - k]) for k in range(K)}


def qubo_energy(Q: QUBO, assignment: Dict[str, int]) -> float:
    """Evaluate the original QUBO at a 0/1 assignment (for validation/selection)."""
    e = 0.0
    for (a, b), w in Q.items():
        e += w * assignment[a] * assignment[b] if a != b else w * assignment[a]
    return e
