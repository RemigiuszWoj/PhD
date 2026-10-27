"""Export the QAOA window circuits as neutral JSON gate lists (track A).

For every first-move window of tai20_5/10/20, instances 0-3 (identity start,
w = 17 linear / w = 6 dense, overlap 0.5 -- the E4 windows of the ICAISC
study) and p = 1..3, the circuit is built exactly as ``solve.py`` builds it
and compiled by qiskit-iqm to the fake Sirius (r = PRX, cz, move; the
resonator is an extra qubit). Two gate lists are written per circuit:

  logical   h, rz, rzz, rx on K qubits (what ``ideal`` simulates)
  compiled  r(theta, phi), cz, move on the active physical qubits + COMPR1,
            with the calibration name of every qubit for the noise lookup

Idle physical qubits are dropped (no gates, no noise in S1-S3), so a K-variable
window simulates K + 1 qubits. ``measure`` maps a circuit qubit to a classical
bit; classical bit k is logical qubit k = ``variables[k]``.

Run from .venv-iqm (iqm-client):
    PYTHONHASHSEED=0 PYTHONPATH=. .venv-iqm/bin/python -m src.simulation.export
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from src.experiments.qaoa_depth_study import H2_TABLE, REPO, TABLES, _noisy_windows, energies

OUT = REPO / "articles" / "annsim_simulators" / "measurements" / "circuits_tai20.json.gz"
P_RANGE = (1, 2, 3)


def _gate_list(qc, keep=None):
    """[[name, [qubits], *params], ...] and [[qubit, clbit], ...] of a circuit.

    ``keep`` renumbers circuit qubits (old index -> new index); qubits outside
    it must carry no operation. The index of a qubit is its position in
    ``qc.qubits`` (what Aer and the noise model use): on circuits returned by
    ``transpile_to_IQM``, ``qc.find_bit(q).index`` can disagree with it (the
    resonator came out as 0 and every qubit shifted by one).
    """
    pos = {q: k for k, q in enumerate(qc.qubits)}
    cpos = {c: k for k, c in enumerate(qc.clbits)}
    ops, meas = [], []
    for inst in qc.data:
        name = inst.operation.name
        qs = [pos[q] for q in inst.qubits]
        if keep is not None:
            qs = [keep[q] for q in qs]
        if name == "barrier":
            continue
        if name == "measure":
            meas.append([qs[0], cpos[inst.clbits[0]]])
            continue
        ops.append([name, qs, *[float(x) for x in inst.operation.params]])
    return ops, meas


def compile_window(Q, table: str, nb: str, p: int, fake):
    from iqm.qiskit_iqm import transpile_to_IQM
    from src.experiments.qaoa_depth_study import _circuit

    variables, qc = _circuit(Q, table, nb, p)
    t0 = time.time()
    tc = transpile_to_IQM(qc, fake, optimization_level=1, seed_transpiler=0)
    compile_s = time.time() - t0
    pos = {q: k for k, q in enumerate(tc.qubits)}
    used = sorted({pos[q] for inst in tc.data if inst.operation.name != "barrier" for q in inst.qubits})
    keep = {q: k for k, q in enumerate(used)}
    names = [fake.index_to_qubit_name(q) if q < fake.num_qubits else "COMPR1" for q in used]
    l_ops, l_meas = _gate_list(qc)
    c_ops, c_meas = _gate_list(tc, keep)
    ops = dict(tc.count_ops())
    return variables, {
        "logical": {"n": qc.num_qubits, "ops": l_ops, "measure": l_meas},
        "compiled": {"n": len(used), "names": names, "ops": c_ops, "measure": c_meas,
                     "cz": ops.get("cz", 0), "move": ops.get("move", 0), "r": ops.get("r", 0),
                     "depth": tc.depth(), "compile_s": compile_s},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    from src.neighborhoods.gate_qaoa.fake_sirius import fake_sirius, latest_snapshot

    fake = fake_sirius(mode="calibrated")
    rows = []
    for f, i, nb, wi, Q in _noisy_windows():
        variables = sorted({v for pair in Q for v in pair})
        E = energies(Q, variables)
        if E.min() >= 0:              # no improving move: QAOA cannot change the outcome
            continue
        table = H2_TABLE[nb]
        for p in P_RANGE:
            if str(p) not in json.loads((REPO / "data" / TABLES[table]).read_text())[nb]:
                continue
            var2, circ = compile_window(Q, table, nb, p, fake)
            assert var2 == variables
            rows.append({"key": f"{f}|{i}|{nb}|{wi}|{table}|{p}", "file": f, "inst": i, "neigh": nb,
                         "window": wi, "table": table, "p": p, "K": len(variables),
                         "variables": variables, "Q": [[a, b, w] for (a, b), w in Q.items()],
                         "E_min": float(E.min()), **circ})
            print(f"{rows[-1]['key']}: K={len(variables)} cz={circ['compiled']['cz']} "
                  f"move={circ['compiled']['move']} r={circ['compiled']['r']}", flush=True)
    meta = {"snapshot": latest_snapshot().name, "compiler": "qiskit-iqm transpile_to_IQM, "
            "optimization_level=1, seed_transpiler=0", "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt") as fh:
        json.dump({"_meta": meta, "circuits": rows}, fh)
    print(f"{len(rows)} circuits -> {args.out}")


if __name__ == "__main__":
    main()
