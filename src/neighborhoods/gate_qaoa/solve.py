"""Runtime QAOA solver for QUBO neighborhood evaluation.

``solve_qaoa(Q, neighborhood, p, backend, angles)`` mirrors
``common.solve_qubo``: it takes a QUBO dict and returns ``{"x0": 0/1, ...}``.
This is the gate-model analog of the single annealer call -- one circuit
execution per move. Backends:

  * ``"statevector"`` : exact noiseless simulation (calibration / validation).
  * ``"aer_noisy"``   : noisy simulation (requires ``qiskit-aer``).
  * ``"ibm"``         : real IBM Heron QPU via Qiskit Runtime.

Angles come from the frozen calibration table ``data/qaoa_angles.json`` keyed
by ``(neighborhood, p)``, or may be passed explicitly as ``(gammas, betas)``.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit.quantum_info import Statevector

from .circuit import (
    QUBO,
    bitstring_to_assignment,
    build_qaoa_circuit,
    normalize_ising,
    qubo_to_ising,
)

_ANGLES_PATH = Path(__file__).resolve().parents[3] / "data" / "qaoa_angles.json"
_ANGLES_CACHE: Optional[dict] = None

# Default IBM Heron r2 target (see project notes).
_IBM_BACKEND = "ibm_fez"


def _load_angles(neighborhood: str, p: int) -> Tuple[List[float], List[float]]:
    global _ANGLES_CACHE
    if _ANGLES_CACHE is None:
        if not _ANGLES_PATH.exists():
            raise FileNotFoundError(
                f"QAOA angle table not found at {_ANGLES_PATH}. Run the angle "
                "calibration (experiments/qaoa_calibrate_angles.py) first, or "
                "pass angles=(gammas, betas) explicitly."
            )
        _ANGLES_CACHE = json.loads(_ANGLES_PATH.read_text())
    try:
        entry = _ANGLES_CACHE[neighborhood][str(p)]
    except KeyError as exc:
        raise KeyError(
            f"No calibrated angles for neighborhood={neighborhood!r}, p={p}."
        ) from exc
    return entry["gamma"], entry["beta"]


def _top_statevector(K, hn, Jn, gammas, betas) -> str:
    qc = build_qaoa_circuit(K, hn, Jn, gammas, betas)
    probs = Statevector(qc).probabilities_dict()
    return max(probs, key=probs.get)


def _top_shots(K, hn, Jn, gammas, betas, backend: str, shots: int) -> str:
    """Build a measured circuit and return the most frequent bitstring."""
    qc = build_qaoa_circuit(K, hn, Jn, gammas, betas, measure=True)

    if backend == "aer_noisy":
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # aer is optional
            raise ImportError(
                "backend='aer_noisy' requires qiskit-aer (pip install qiskit-aer)."
            ) from exc
        sim = AerSimulator()
        from qiskit import transpile
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        return max(counts, key=counts.get)

    # backend == "ibm": real IBM Heron QPU
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    token = os.environ.get("IBM_TOKEN")
    if not token:
        raise ValueError("backend='ibm' requires IBM_TOKEN in the environment (.env).")
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=os.environ.get("IBM_CRN"),
    )
    hw = service.backend(os.environ.get("IBM_BACKEND", _IBM_BACKEND))
    isa = generate_preset_pass_manager(optimization_level=1, backend=hw).run(qc)
    result = SamplerV2(mode=hw).run([isa], shots=shots).result()
    counts = result[0].data.meas.get_counts()
    return max(counts, key=counts.get)


def solve_qaoa(
    Q: QUBO,
    neighborhood: str,
    p: int = 1,
    backend: str = "statevector",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
) -> Dict[str, int]:
    """Solve a QUBO with fixed-angle QAOA; returns ``{"x_k": 0/1}``.

    Same contract and return shape as ``common.solve_qubo`` so a neighborhood
    can swap the annealer for QAOA without changing its map-back.
    """
    if not Q:
        return {}
    variables, h, J, _ = qubo_to_ising(Q)
    K = len(variables)
    hn, Jn, _ = normalize_ising(h, J)

    gammas, betas = angles if angles is not None else _load_angles(neighborhood, p)
    if len(gammas) != p or len(betas) != p:
        raise ValueError(
            f"angle schedule length {len(gammas)}/{len(betas)} does not match p={p}."
        )

    if backend == "statevector":
        top = _top_statevector(K, hn, Jn, gammas, betas)
    elif backend in ("aer_noisy", "ibm"):
        top = _top_shots(K, hn, Jn, gammas, betas, backend, shots)
    else:
        raise ValueError(f"unknown backend {backend!r} (statevector|aer_noisy|ibm).")

    return bitstring_to_assignment(top, variables)
