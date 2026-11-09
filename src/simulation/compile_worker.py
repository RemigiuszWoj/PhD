"""Compile worker for runs that simulate inside ILS (backend.py, t5_ils.py).

The simulators live in .venv-annsim (qiskit 2.6), the compiler of the hardware
runs and of track A in .venv-iqm (iqm-client 35, qiskit < 2.2), so a run in
.venv-annsim keeps one of these workers open and sends it the logical circuits
of every move. One JSON line in (``{"logical": gate list}``), one out (the
compiled gate list of export.compile_qc). The circuit is rebuilt with
``measure_all`` as build_qaoa_circuit makes it, so the compilation is the same
as for the hardware runs and the export (checked on the export's windows).

Started by backend.py as:
    PYTHONPATH=. .venv-iqm/bin/python -m src.simulation.compile_worker
"""
from __future__ import annotations

import json
import sys


def rebuild(logical: dict):
    """Measured logical circuit from its gate list, laid out as build_qaoa_circuit lays it out."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(logical["n"])
    for name, qs, *par in logical["ops"]:
        if name == "h":
            qc.h(qs[0])
        elif name == "rz":
            qc.rz(par[0], qs[0])
        elif name == "rzz":
            qc.rzz(par[0], qs[0], qs[1])
        elif name == "rx":
            qc.rx(par[0], qs[0])
        else:
            raise ValueError(f"gate {name} in a logical circuit")
    if logical["measure"] != [[k, k] for k in range(logical["n"])]:
        raise ValueError("logical circuits measure qubit k into bit k")
    qc.measure_all()
    return qc


def main():
    from src.neighborhoods.gate_qaoa.fake_sirius import fake_sirius

    from .export import compile_qc

    fake = fake_sirius(mode="calibrated")
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        try:
            out = {"compiled": compile_qc(rebuild(req["logical"]), fake)["compiled"]}
        except Exception as exc:                      # reported to the caller, which raises
            out = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
