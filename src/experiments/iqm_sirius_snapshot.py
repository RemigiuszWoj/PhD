"""Snapshot of the IQM Sirius calibration for the fake-Sirius noise model.

Reads the dynamic architecture and the calibration quality metrics of the
current calibration set from IQM Resonance and writes them to
``data/iqm_sirius_calibration_<YYYYMMDD>.json``. These are plain API reads:
no circuit is submitted, so no credits are used. Only operational qubits
(those with a calibrated ``prx``) are kept; the device lists 24, of which 16
carry gate calibrations. Units: seconds for T1/T2 and gate durations.

Usage (IQM_TOKEN in .env, run from .venv-iqm):
    PYTHONPATH=. .venv-iqm/bin/python -m src.experiments.iqm_sirius_snapshot
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

from iqm.iqm_client import IQMClient  # noqa: E402  (reads IQM_TOKEN itself)


def snapshot(computer: str = "sirius") -> dict:
    client = IQMClient(os.environ.get("IQM_SERVER", "https://resonance.iqm.tech"),
                       quantum_computer=computer)
    dqa = client.get_dynamic_quantum_architecture()
    qm = client.get_calibration_quality_metrics(dqa.calibration_set_id)

    def gate(name):
        """{locus: {impl, fidelity, duration}} for every implementation of a gate."""
        out = {}
        for impl, info in dqa.gates[name].implementations.items():
            for locus in info.loci:
                out["|".join(locus)] = {
                    "impl": impl,
                    "default": impl == dqa.gates[name].default_implementation,
                    "fidelity": qm.get_gate_fidelity(name, impl, locus),
                    "duration": qm.get_gate_duration(name, impl, locus),
                }
        return out

    prx = gate("prx")
    qubits = sorted({k for k in prx}, key=lambda q: int(q[2:]))
    resonators = list(dqa.computational_resonators)
    t1, t2 = qm.get_coherence_times(qubits + resonators)
    impl = dqa.gates["measure"].default_implementation
    readout = {}
    for q in qubits:
        err = qm.get_measure_errors("measure", impl, (q,))
        readout[q] = {"0": err[0], "1": err[1]}   # P(read 1 | 0), P(read 0 | 1)

    return {
        "computer": computer,
        "calibration_set_id": str(dqa.calibration_set_id),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qubits": qubits,
        "computational_resonators": resonators,
        "t1": t1,
        "t2": t2,
        "prx": prx,
        "cz": gate("cz"),
        "move": gate("move"),
        "readout": readout,
    }


def main():
    data = snapshot()
    day = data["fetched_utc"][:10].replace("-", "")
    out = REPO / "data" / f"iqm_sirius_calibration_{day}.json"
    out.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"{len(data['qubits'])} qubits, calibration set {data['calibration_set_id']} -> {out}")


if __name__ == "__main__":
    main()
