"""Runtime QAOA solver for QUBO neighborhood evaluation.

``solve_qaoa(Q, neighborhood, p, backend, angles)`` mirrors
``common.solve_qubo``: it takes a QUBO dict and returns ``{"x0": 0/1, ...}``.
This is the gate-model analog of the single annealer call -- one circuit
execution per move. Backends:

  * ``"ibm"``       : real IBM Heron QPU via Qiskit Runtime (default).
  * ``"aer_noisy"`` : noisy simulation (requires ``qiskit-aer``).

Experiments run on hardware only; there is no noiseless-simulator backend.
Offline angle calibration (``angles.py``) is a separate, one-off step and is
the only place a statevector is evaluated.

Readout (standard QAOA): the circuit prepares a distribution over bitstrings;
we draw ``shots`` samples and return the sample with the lowest *original*
QUBO energy. Reading out the single most-probable bitstring instead is wrong:
at low p the argmax basis state is typically the all-zeros (do-nothing) state
or a high-energy, many-bit state, neither of which is the QUBO optimum the
distribution actually concentrates on.

Angles come from the frozen calibration table ``data/qaoa_angles.json`` keyed
by ``(neighborhood, p)``, or may be passed explicitly as ``(gammas, betas)``.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


_IBM_CACHE = None


def _get_ibm():
    """Cached (backend, pass manager); building these per move is slow."""
    global _IBM_CACHE
    if _IBM_CACHE is None:
        from qiskit_ibm_runtime import QiskitRuntimeService
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
        _IBM_CACHE = (hw, generate_preset_pass_manager(optimization_level=1, backend=hw))
    return _IBM_CACHE


def _device_counts_batch(circuits, backend: str, shots: int) -> List[Dict[str, int]]:
    """Run all circuits and return one shot-count histogram per circuit.

    On ``ibm`` the whole list goes out as a single SamplerV2 job, so the queue is
    paid once per batch rather than once per circuit.
    """
    if backend == "aer_noisy":
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # aer is optional
            raise ImportError(
                "backend='aer_noisy' requires qiskit-aer (pip install qiskit-aer)."
            ) from exc
        from qiskit import transpile
        sim = AerSimulator()
        res = sim.run(transpile(circuits, sim), shots=shots).result()
        return [res.get_counts(i) for i in range(len(circuits))]

    # backend == "ibm": real IBM Heron QPU
    from qiskit_ibm_runtime import SamplerV2

    hw, pm = _get_ibm()
    result = SamplerV2(mode=hw).run(pm.run(circuits), shots=shots).result()
    return [result[i].data.meas.get_counts() for i in range(len(circuits))]


def _prepare(Q: QUBO, neighborhood: str, p: int, angles):
    """Shared setup: Ising mapping, normalization and the angle schedule."""
    variables, h, J, _ = qubo_to_ising(Q)
    hn, Jn, _ = normalize_ising(h, J)
    gammas, betas = angles if angles is not None else _load_angles(neighborhood, p)
    if len(gammas) != p or len(betas) != p:
        raise ValueError(
            f"angle schedule length {len(gammas)}/{len(betas)} does not match p={p}."
        )
    return variables, len(variables), hn, Jn, gammas, betas


def _pick_lowest(Q: QUBO, variables, counts) -> Dict[str, int]:
    """Energy-based readout: lowest original-QUBO-energy sample."""
    best = min(counts, key=lambda bs: qubo_energy(Q, bitstring_to_assignment(bs, variables)))
    return bitstring_to_assignment(best, variables)


def solve_qaoa_batch(
    Qs: Sequence[QUBO],
    neighborhood: str,
    p: int = 1,
    backend: str = "ibm",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
) -> List[Dict[str, int]]:
    """Solve several QUBOs at once; returns one assignment per input QUBO.

    Every circuit of the batch is submitted in a single job, so the queue is
    paid once per batch instead of once per QUBO -- the windows of one move
    become one submission.
    """
    out: List[Dict[str, int]] = [{} for _ in Qs]
    prepared = {i: _prepare(Q, neighborhood, p, angles) for i, Q in enumerate(Qs) if Q}

    if backend not in ("aer_noisy", "ibm"):
        raise ValueError(f"unknown backend {backend!r} (ibm|aer_noisy).")

    idx = list(prepared)
    circuits = [
        build_qaoa_circuit(prepared[i][1], prepared[i][2], prepared[i][3],
                           prepared[i][4], prepared[i][5], measure=True)
        for i in idx
    ]
    for i, counts in zip(idx, _device_counts_batch(circuits, backend, shots)):
        out[i] = _pick_lowest(Qs[i], prepared[i][0], counts)
    return out


def solve_qaoa(
    Q: QUBO,
    neighborhood: str,
    p: int = 1,
    backend: str = "ibm",
    angles: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    shots: int = 4096,
) -> Dict[str, int]:
    """Solve a QUBO with fixed-angle QAOA; returns ``{"x_k": 0/1}``.

    Same contract and return shape as ``common.solve_qubo`` so a neighborhood
    can swap the annealer for QAOA without changing its map-back. The returned
    assignment is the lowest-QUBO-energy sample among ``shots`` draws from the
    QAOA distribution. This is the single-QUBO case of ``solve_qaoa_batch``.
    """
    if not Q:
        return {}
    return solve_qaoa_batch([Q], neighborhood, p=p, backend=backend,
                            angles=angles, shots=shots)[0]
