"""Fake IQM Sirius (16-qubit star with a computational resonator).

iqm-client ships fake backends for Adonis, Apollo, Aphrodite and Deneb, but no
16-qubit star, so this builds one from a calibration snapshot of the real
device (``src/experiments/iqm_sirius_snapshot.py``). The architecture is the
operational part of Sirius: every qubit coupled to the resonator COMPR1 by
``cz`` and ``move``, exactly the loci the device calibrates.

Noise follows IQMFakeBackend: per gate a thermal relaxation channel (T1, T2,
gate duration) composed with a depolarizing channel, plus per-qubit readout
errors. Two modes:

  * ``"thermal"``: the measured average gate fidelity already contains
    relaxation, so the depolarizing part gets only the remainder,
    ``1 - F_dep = (1 - F_measured) - (1 - F_relaxation)`` (clipped at 0). On
    Sirius relaxation alone exceeds the measured infidelity on most loci
    (resonator T2 ~3 us), so this mode is ~25-35 % more pessimistic per gate
    than the device reports -- an upper bound on the noise.
  * ``"calibrated"`` (default): no relaxation on gates; the whole measured
    infidelity goes into the depolarizing channel, so every gate has exactly
    the fidelity IQM reports.

The channel parameter is ``lambda = d (1 - F_dep) / (d - 1)``. Gate durations
are per gate name in IQMFakeBackend, so the median over loci is used. Neither
mode adds idle noise: Aer applies errors on gates only.
"""
import json
from pathlib import Path
from statistics import median
from typing import Optional

from iqm.qiskit_iqm.fake_backends.iqm_fake_backend import IQMErrorProfile, IQMFakeBackend
from iqm.station_control.interface.models import StaticQuantumArchitecture
from qiskit.quantum_info import average_gate_fidelity
from qiskit_aer.noise.errors import thermal_relaxation_error

_DATA = Path(__file__).resolve().parents[3] / "data"
_NS = 1e9  # snapshot is in seconds, IQMFakeBackend works in nanoseconds
_NO_RELAXATION_NS = 1e15  # T1 = T2 so long that thermal_relaxation_error is the identity


def latest_snapshot() -> Path:
    files = sorted(_DATA.glob("iqm_sirius_calibration_*.json"))
    if not files:
        raise FileNotFoundError(
            "no Sirius calibration snapshot in data/; run "
            "src/experiments/iqm_sirius_snapshot.py first.")
    return files[-1]


def _depolarizing(fidelity: float, relaxation, d: int) -> float:
    """Depolarizing parameter left over after the relaxation part of the infidelity."""
    remainder = (1.0 - fidelity) - (1.0 - average_gate_fidelity(relaxation))
    return d * max(0.0, remainder) / (d - 1)


def fake_sirius(snapshot: Optional[Path] = None, mode: str = "calibrated") -> IQMFakeBackend:
    if mode not in ("calibrated", "thermal"):
        raise ValueError(f"mode={mode!r} (calibrated|thermal).")
    snap = json.loads(Path(snapshot or latest_snapshot()).read_text())
    qubits = snap["qubits"]
    res = snap["computational_resonators"][0]
    if mode == "thermal":
        t1 = {c: v * _NS for c, v in snap["t1"].items()}
        t2 = {c: v * _NS for c, v in snap["t2"].items()}
    else:
        t1 = {c: _NO_RELAXATION_NS for c in snap["t1"]}
        t2 = {c: _NO_RELAXATION_NS for c in snap["t2"]}
    dur = {g: median(e["duration"] for e in snap[g].values()) * _NS
           for g in ("prx", "cz", "move")}

    def relax(component, t):
        return thermal_relaxation_error(t1[component], t2[component], t)

    prx = {q: _depolarizing(snap["prx"][q]["fidelity"], relax(q, dur["prx"]), 2) for q in qubits}
    two = {}
    for g in ("cz", "move"):
        two[g] = {}
        for locus, e in snap[g].items():
            q, r = locus.split("|")
            channel = relax(q, dur[g]).tensor(relax(r, dur[g]))
            two[g][(q, r)] = _depolarizing(e["fidelity"], channel, 4)

    architecture = StaticQuantumArchitecture(
        dut_label=f"sirius-{snap['calibration_set_id'][:8]}",
        qubits=qubits,
        computational_resonators=[res],
        connectivity=[(res, q) for q in qubits],
    )
    profile = IQMErrorProfile(
        t1s=t1, t2s=t2,
        single_qubit_gate_depolarizing_error_parameters={"prx": prx},
        two_qubit_gate_depolarizing_error_parameters=two,
        single_qubit_gate_durations={"prx": dur["prx"]},
        two_qubit_gate_durations={"cz": dur["cz"], "move": dur["move"]},
        readout_errors=snap["readout"],
        name=f"Sirius-{mode}",
    )
    return IQMFakeBackend(architecture, profile, name=f"IQMFakeSirius_{mode}")
