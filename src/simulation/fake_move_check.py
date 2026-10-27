"""How much does the missing MOVE noise of the fake Sirius matter?

``IQMFakeBackend.run`` replaces every ``move`` by an unnamed ``UnitaryGate``
before Aer runs the circuit, so the "move" errors of its noise model are never
applied (check: 200 MOVEs leave P(1) of |1> unchanged, 200 CZs depolarize it).
For the given exported circuits (export.py) this runs, all with Aer density
matrices (exact, readout applied):

  fake      the fake backend's noise model as ICAISC E4/E5n used it (MOVE noiseless)
  fake+mv   the same noise model with MOVE kept as a gate named "move" (labelled
            unitary), so its errors are applied
  ours      the S2 channels of noise.py (should equal fake+mv: same parameters)

and records P_opt and the TVD between the three distributions.

Run from .venv-iqm:
    PYTHONPATH=. .venv-iqm/bin/python -m src.simulation.fake_move_check --keys ...
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .noise import REPO

CIRCUITS = REPO / "articles" / "annsim_simulators" / "measurements" / "circuits_tai20.json.gz"
OUT = CIRCUITS.with_name("fake_sirius_move_check.jsonl")


def to_qiskit(c, move_label: str | None):
    """Exported compiled circuit on the fake backend's 17 indices (16 qubits + COMPR1)."""
    from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY, MoveGate
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate, UnitaryGate

    from src.neighborhoods.gate_qaoa.fake_sirius import fake_sirius

    fake = fake_sirius(mode="calibrated")
    phys = [fake.qubit_name_to_index(n) for n in c["names"]]
    qc = QuantumCircuit(17)
    for name, qs, *par in c["ops"]:
        q = [phys[x] for x in qs]
        if name == "r":
            qc.append(RGate(par[0], par[1]), q)
        elif name == "cz":
            qc.cz(*q)
        elif name == "move":
            qc.append(MoveGate() if move_label is None else UnitaryGate(MOVE_GATE_UNITARY, label=move_label), q)
    meas = [phys[q] for q, _ in sorted(c["measure"], key=lambda m: m[1])]
    return fake, qc, meas


def dm_probs(qc, meas, noise_model):
    from qiskit_aer import AerSimulator

    qc = qc.copy()
    qc.save_probabilities(qubits=meas)
    r = AerSimulator(method="density_matrix", noise_model=noise_model).run(qc).result()
    return np.asarray(r.data(0)["probabilities"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    from iqm.qiskit_iqm.iqm_transpilation import IQMReplaceGateWithUnitaryPass
    from iqm.qiskit_iqm.move_gate import MOVE_GATE_UNITARY

    from .cell import energies, readout_errors
    from .noise import NoiseParams, apply_readout_probs

    with gzip.open(CIRCUITS, "rt") as fh:
        cs = {c["key"]: c for c in json.load(fh)["circuits"]}
    ref_dir = CIRCUITS.parent / "refs"
    with args.out.open("a") as out:
        for key in args.keys:
            c = cs[key]
            fake, qc, meas = to_qiskit(c["compiled"], None)
            # exactly what fake.run does: MOVE -> unnamed unitary, then the fake's noise model
            plain = IQMReplaceGateWithUnitaryPass("move", MOVE_GATE_UNITARY)(qc)
            p_fake = dm_probs(plain, meas, fake.noise_model)
            _, qc_mv, _ = to_qiskit(c["compiled"], "move")
            p_mv = dm_probs(qc_mv, meas, fake.noise_model)
            # Aer applies readout errors to measurements only, not to saved probabilities: add the same
            # readout channel the S2 reference has (same snapshot values as the fake backend's)
            errors = readout_errors(c, NoiseParams("S2"))
            p_fake, p_mv = apply_readout_probs(p_fake, errors), apply_readout_probs(p_mv, errors)
            E = energies(c)
            opt = np.abs(E - E.min()) < 1e-9
            row = {"key": key, "K": c["K"], "cz": c["compiled"]["cz"], "move": c["compiled"]["move"],
                   "P_opt_fake": float(p_fake[opt].sum()), "P_opt_fake_mv": float(p_mv[opt].sum()),
                   "tvd_fake_vs_mv": float(0.5 * np.abs(p_fake - p_mv).sum())}
            rp = ref_dir / (key.replace("|", "_") + "_S2.npy")
            if rp.exists():
                ours = np.load(rp)
                row.update({"P_opt_ours_S2": float(ours[opt].sum()),
                            "tvd_ours_vs_mv": float(0.5 * np.abs(ours - p_mv).sum()),
                            "tvd_ours_vs_fake": float(0.5 * np.abs(ours - p_fake).sum())})
            rp1 = ref_dir / (key.replace("|", "_") + "_S1.npy")
            if rp1.exists():
                row["P_opt_ideal"] = float(np.load(rp1)[opt].sum())
            out.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
