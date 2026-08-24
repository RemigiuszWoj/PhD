"""Runtime QAOA solver for QUBO neighborhood evaluation.

``solve_qaoa(Q, neighborhood, p, backend, angles)`` mirrors
``common.solve_qubo``: it takes a QUBO dict and returns ``{"x0": 0/1, ...}``.
This is the gate-model analog of the single annealer call -- one circuit
execution per move. Backends:

  * ``"statevector"`` : exact noiseless simulation (calibration / validation).
  * ``"aer_noisy"``   : noisy simulation (requires ``qiskit-aer``).
  * ``"ibm"``         : real IBM Heron QPU via Qiskit Runtime.

Readout (standard QAOA): the circuit prepares a distribution over bitstrings;
we draw ``shots`` samples and return the sample with the lowest *original*
QUBO energy -- exactly what one does on hardware (finite shots, pick the best).
The statevector backend samples from the exact distribution; the device
backends sample from the hardware/noisy simulator. Reading out the single
most-probable bitstring instead is wrong: at low p the argmax basis state is
typically the all-zeros (do-nothing) state or a high-energy, many-bit state,
neither of which is the QUBO optimum the distribution actually concentrates on.

Sampling is seeded from the QUBO content, so a given move is reproducible
across runs (the same sub-QUBO always yields the same selection on the
noiseless simulator).

Angles come from the frozen calibration table ``data/qaoa_angles.json`` keyed
by ``(neighborhood, p)``, or may be passed explicitly as ``(gammas, betas)``.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import Statevector

from .circuit import (
    QUBO,
    bitstring_to_assignment,
    build_qaoa_circuit,
    normalize_ising,
    qubo_energy,
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


def _qubo_seed(Q: QUBO) -> int:
    """Deterministic seed derived from the QUBO content.

    Makes a move's statevector sampling reproducible: the same sub-QUBO always
    produces the same sampled selection on the noiseless simulator.
    """
    items = sorted((f"{a}|{b}", round(float(w), 6)) for (a, b), w in Q.items())
    return int(hashlib.md5(repr(items).encode()).hexdigest()[:8], 16)


def _statevector_counts(K, hn, Jn, gammas, betas, shots: int, seed: int) -> Dict[str, int]:
    """Sample ``shots`` bitstrings from the exact QAOA distribution."""
    qc = build_qaoa_circuit(K, hn, Jn, gammas, betas)
    probs = Statevector(qc).probabilities_dict()
    keys = list(probs)
    p = np.fromiter((probs[k] for k in keys), dtype=float)
    p /= p.sum()
    draws = np.random.default_rng(seed).choice(len(keys), size=shots, p=p)
    vals, cnts = np.unique(draws, return_counts=True)
    return {keys[int(i)]: int(c) for i, c in zip(vals, cnts)}


def _device_counts(K, hn, Jn, gammas, betas, backend: str, shots: int) -> Dict[str, int]:
    """Run a measured circuit and return the full shot-count histogram."""
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
        return sim.run(transpile(qc, sim), shots=shots).result().get_counts()

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
    return result[0].data.meas.get_counts()


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
    can swap the annealer for QAOA without changing its map-back. The returned
    assignment is the lowest-QUBO-energy sample among ``shots`` draws from the
    QAOA distribution.
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
        counts = _statevector_counts(K, hn, Jn, gammas, betas, shots, _qubo_seed(Q))
    elif backend in ("aer_noisy", "ibm"):
        counts = _device_counts(K, hn, Jn, gammas, betas, backend, shots)
    else:
        raise ValueError(f"unknown backend {backend!r} (statevector|aer_noisy|ibm).")

    # Energy-based readout: pick the lowest original-QUBO-energy sampled bitstring.
    best = min(
        counts,
        key=lambda bs: qubo_energy(Q, bitstring_to_assignment(bs, variables)),
    )
    return bitstring_to_assignment(best, variables)
