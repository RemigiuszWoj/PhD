"""QAOA circuit construction for PFSP QUBO neighborhoods.

Builds QAOA circuits of depth p using only CZ gates (native Willow gates).

Gate decomposition:
    J_ij * Z_i * Z_j  →  CZ -- R_Z(2γ J_ij) -- CZ   (2 CZ gates)
    h_i  * Z_i        →  R_Z(2γ h_i)                 (1 single-qubit gate)
    Mixer             →  R_X(2β) per qubit
"""

from typing import List, Tuple, Dict
import numpy as np
import cirq


def build_qaoa_circuit(
    h: List[float],
    J: Dict[Tuple[int, int], float],
    gammas: List[float],
    betas: List[float],
) -> cirq.Circuit:
    """Build QAOA circuit for Ising Hamiltonian H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j.

    Uses only CZ + R_Z + R_X gates (native Willow gate set).

    Args:
        h: Linear coefficients h_i (length K)
        J: Quadratic coefficients {(i,j): J_ij} for i < j
        gammas: Cost unitary angles, one per QAOA layer
        betas: Mixer unitary angles, one per QAOA layer

    Returns:
        cirq.Circuit ready for simulation or QPU execution
    """
    assert len(gammas) == len(betas), "gammas and betas must have same length"
    p = len(gammas)
    K = len(h)

    qubits = cirq.LineQubit.range(K)
    circuit = cirq.Circuit()

    # Initial state: |+>^K = H^K |0>^K
    circuit.append(cirq.H(q) for q in qubits)

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Cost unitary U_C(gamma) = exp(-i gamma H_C)
        # Z_i Z_j term: CZ -- R_Z(2 gamma J_ij) -- CZ
        for (i, j), jval in J.items():
            if abs(jval) < 1e-12:
                continue
            circuit.append([
                cirq.CZ(qubits[i], qubits[j]),
                cirq.rz(2 * gamma * jval)(qubits[j]),
                cirq.CZ(qubits[i], qubits[j]),
            ])

        # Z_i term: R_Z(2 gamma h_i)
        for i, hi in enumerate(h):
            if abs(hi) < 1e-12:
                continue
            circuit.append(cirq.rz(2 * gamma * hi)(qubits[i]))

        # Mixer unitary U_B(beta) = exp(-i beta Σ X_i)
        circuit.append(cirq.rx(2 * beta)(q) for q in qubits)

    # Measurement
    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit


def circuit_stats(
    h: List[float],
    J: Dict[Tuple[int, int], float],
    p: int,
) -> Dict[str, int]:
    """Compute circuit resource estimates without building full circuit.

    Args:
        h: Linear coefficients
        J: Quadratic coefficients
        p: QAOA depth

    Returns:
        Dict with n_qubits, cz_per_layer, total_cz, accumulated_error_pct
    """
    K = len(h)
    nonzero_J = sum(1 for v in J.values() if abs(v) > 1e-12)
    nonzero_h = sum(1 for v in h if abs(v) > 1e-12)

    cz_per_layer = 2 * nonzero_J
    rz_per_layer = nonzero_h + nonzero_J
    rx_per_layer = K

    return {
        "n_qubits": K,
        "cz_per_layer": cz_per_layer,
        "total_cz": cz_per_layer * p,
        "total_single_qubit": (rz_per_layer + rx_per_layer) * p,
        "accumulated_error_pct": round(
            (1 - (1 - 0.0014) ** (cz_per_layer * p)) * 100, 2
        ),
    }
