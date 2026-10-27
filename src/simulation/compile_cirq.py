"""Track B, Cirq compiler: the logical circuits of export.py compiled by cirq-iqm.

The device is built offline from the Sirius calibration snapshot (a
``DynamicQuantumArchitecture`` with prx, cz, move and measure loci), so no
connection to IQM Resonance is made. Pipeline as in the cirq-iqm user guide:
``decompose_circuit`` -> ``route_circuit`` (RouteCQC on the qubit graph where
every pair sharing the resonator is connected, then ``transpile_insert_moves``)
-> ``simplify_circuit``. The result is written in the gate-list format of
export.py (r, cz, move; names for the noise lookup), so every engine can run
it. CZ = CZPowGate(exponent=1); PhasedXPowGate(t, p) = R(pi t, pi p) up to a
global phase; MOVE stays MOVE; a ZPowGate left by simplify_circuit becomes
``rz`` (virtual on IQM, so no noise channel follows it).

Run from .venv-annsim-cirq (iqm-client[cirq] 36):
    PYTHONPATH=. .venv-annsim-cirq/bin/python -m src.simulation.compile_cirq [--keys ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .noise import REPO, latest_snapshot

CIRCUITS = REPO / "articles" / "annsim_simulators" / "measurements" / "circuits_tai20.json.gz"
OUT = CIRCUITS.with_name("circuits_tai20_cirq.json.gz")


def sirius_device(snapshot: Path | None = None):
    """IQMDevice of Sirius from the calibration snapshot (default implementations only)."""
    from iqm.cirq_iqm.devices.iqm_device import IQMDevice
    from iqm.cirq_iqm.devices.iqm_device_metadata import IQMDeviceMetadata
    from iqm.iqm_client import DynamicQuantumArchitecture, GateImplementationInfo, GateInfo

    snap = json.loads(Path(snapshot or latest_snapshot()).read_text())
    qubits, res = snap["qubits"], snap["computational_resonators"]

    def gate(loci, impl):
        return GateInfo(implementations={impl: GateImplementationInfo(loci=tuple(loci))},
                        default_implementation=impl, override_default_implementation={})

    two = [tuple(k.split("|")) for k in snap["cz"]]
    dqa = DynamicQuantumArchitecture(
        calibration_set_id=uuid.UUID(snap["calibration_set_id"]),
        qubits=qubits, computational_resonators=res,
        gates={"prx": gate([(q,) for q in qubits], "drag_crf"),
               "cz": gate(two, "crf_crf"),
               "move": gate([tuple(k.split("|")) for k in snap["move"]], "crf_crf"),
               "measure": gate([(q,) for q in qubits], "constant")})
    return IQMDevice(IQMDeviceMetadata.from_architecture(dqa))


def logical_cirq(circ):
    import cirq

    q = cirq.NamedQubit.range(circ["n"], prefix="v")
    ops = []
    for name, qs, *par in circ["ops"]:
        if name == "h":
            ops.append(cirq.H(q[qs[0]]))
        elif name == "rz":
            ops.append(cirq.rz(par[0])(q[qs[0]]))
        elif name == "rx":
            ops.append(cirq.rx(par[0])(q[qs[0]]))
        elif name == "rzz":
            ops.append(cirq.ZZPowGate(exponent=par[0] / np.pi, global_shift=-0.5)(q[qs[0]], q[qs[1]]))
        else:
            raise ValueError(name)
    ops += [cirq.measure(q[k], key=f"c{c}") for k, c in circ["measure"]]
    return cirq.Circuit(ops)


def compile_one(c, device):
    import cirq
    from iqm.cirq_iqm.iqm_gates import IQMMoveGate
    from iqm.cirq_iqm.optimizers import simplify_circuit

    t0 = time.time()
    decomposed = device.decompose_circuit(logical_cirq(c["logical"]))
    routed, _, _ = device.route_circuit(decomposed)
    final = simplify_circuit(routed, drop_final_rz=True)
    compile_s = time.time() - t0
    used = sorted({qid.name for op in final.all_operations() for qid in op.qubits},
                  key=lambda s: (s.startswith("COMP"), int(s[2:]) if s.startswith("QB") else 0))
    idx = {name: k for k, name in enumerate(used)}
    ops, meas, counts = [], [], {"cz": 0, "move": 0, "r": 0}
    for op in final.all_operations():
        g, qs = op.gate, [idx[x.name] for x in op.qubits]
        if isinstance(g, cirq.MeasurementGate):
            meas.append([qs[0], int(g.key[1:])])
        elif isinstance(g, IQMMoveGate):
            ops.append(["move", qs]); counts["move"] += 1
        elif isinstance(g, cirq.CZPowGate) and abs(g.exponent - 1) < 1e-12:
            ops.append(["cz", qs]); counts["cz"] += 1
        elif isinstance(g, cirq.PhasedXPowGate):
            ops.append(["r", qs, float(g.exponent) * np.pi, float(g.phase_exponent) * np.pi]); counts["r"] += 1
        elif isinstance(g, cirq.ZPowGate):            # virtual Z on IQM: no pulse, no noise
            ops.append(["rz", qs, float(g.exponent) * np.pi]); counts["rz"] = counts.get("rz", 0) + 1
        else:
            raise ValueError(f"unexpected gate in compiled circuit: {op!r}")
    return {"n": len(used), "names": used, "ops": ops, "measure": meas, **counts,
            "depth": len(final), "compile_s": compile_s}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    device = sirius_device()
    with gzip.open(CIRCUITS, "rt") as fh:
        data = json.load(fh)
    rows = []
    for c in data["circuits"]:
        if args.keys and c["key"] not in args.keys:
            continue
        try:
            comp = compile_one(c, device)
        except Exception as exc:
            print(f"{c['key']}: FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({**{k: v for k, v in c.items() if k != "compiled"}, "compiled": comp,
                     "compiled_qiskit": {k: c["compiled"][k] for k in ("cz", "move", "r", "depth")}})
        print(f"{c['key']}: K={c['K']} cirq cz={comp['cz']} move={comp['move']} r={comp['r']} | "
              f"qiskit cz={c['compiled']['cz']} move={c['compiled']['move']} r={c['compiled']['r']}", flush=True)
    meta = {**data["_meta"], "compiler": "cirq-iqm (iqm-client 36) decompose -> route_circuit -> "
            "simplify_circuit, device from the calibration snapshot",
            "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with gzip.open(args.out, "wt") as fh:
        json.dump({"_meta": meta, "circuits": rows}, fh)
    print(f"{len(rows)} circuits -> {args.out}")


if __name__ == "__main__":
    main()
