"""H3: window histograms on IQM Sirius for the ANNSIM study (one run per day).

The first-move windows of tai20_5/10/20, instance 0, with an improving move
(the rows of circuits_tai20.json.gz) at p = 1 and 2: 42 windows, 84 circuits.
Each logical circuit is rebuilt exactly as ``solve.py`` builds it, checked
against the gate list the simulators ran, transpiled to the real Sirius and
sent as one job of 4096 shots per circuit. A row keeps the compiled gate list
that actually ran, the counts, the job id, the job timeline and the calibration
set, so every simulator can replay the same circuit on the same-day snapshot.
Run it on two different days: the second run is the device-vs-device floor.

Take a calibration snapshot first, then (from the annsim-simulators worktree):
    PYTHONPATH=. ../../../.venv-iqm/bin/python -m src.experiments.iqm_sirius_snapshot
    PYTHONHASHSEED=0 PYTHONPATH=. ../../../.venv-iqm/bin/python -m src.simulation.h3 \
        --env ../../../.env [--dry-run] [--max-credits 45]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

from src.experiments.qaoa_depth_study import REPO, _circuit
from src.simulation.export import OUT as CIRCUITS, _gate_list

OUT = REPO / "articles" / "annsim_simulators" / "measurements" / "h3_sirius.jsonl"
P_H3 = (1, 2)
SHOTS = 4096


def _rows():
    rows = json.load(gzip.open(CIRCUITS, "rt"))["circuits"]
    return [r for r in rows if r["inst"] == 0 and r["p"] in P_H3]


def _logical(r):
    """The measured logical circuit of a row, rebuilt and checked against the stored gate list."""
    Q = {(a, b): w for a, b, w in r["Q"]}
    variables, qc = _circuit(Q, r["table"], r["neigh"], r["p"])
    assert variables == r["variables"], r["key"]
    ops, meas = _gate_list(qc)
    assert ops == r["logical"]["ops"] and meas == r["logical"]["measure"], r["key"]
    return qc


def _compiled(tc, hw) -> dict:
    """Gate list of a transpiled circuit on its active qubits, as in export.compile_qc."""
    pos = {q: k for k, q in enumerate(tc.qubits)}
    used = sorted({pos[q] for inst in tc.data if inst.operation.name != "barrier" for q in inst.qubits})
    keep = {q: k for k, q in enumerate(used)}
    names = [hw.index_to_qubit_name(q) if q < hw.num_qubits else "COMPR1" for q in used]
    ops, meas = _gate_list(tc, keep)
    count = dict(tc.count_ops())
    return {"n": len(used), "names": names, "ops": ops, "measure": meas, "cz": count.get("cz", 0),
            "move": count.get("move", 0), "r": count.get("r", 0), "depth": tc.depth()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-credits", type=float, default=45.0)
    ap.add_argument("--rate", type=float, default=0.445,
                    help="credits per circuit (0.445 measured over the ACIIDS campaign)")
    ap.add_argument("--env", type=Path, default=REPO / ".env", help="file with IQM_PWR_TOKEN")
    ap.add_argument("--day", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--max-circuits", type=int, default=None, help="send only the first N of the remaining")
    ap.add_argument("--p", type=int, choices=P_H3, default=None, help="send only circuits of this depth")
    args = ap.parse_args()

    rows = _rows()
    done = set()
    if OUT.exists():
        done = {(d["key"], d["day"]) for d in map(json.loads, OUT.read_text().splitlines())}
    left = [r for r in rows if (r["key"], args.day) not in done and args.p in (None, r["p"])]
    todo = left[:args.max_circuits]
    qcs = [_logical(r) for r in todo]
    est = len(todo) * args.rate
    print(f"h3 {args.day}: {len(rows)} circuits, {len(rows) - len(left)} done, {len(left)} to go, "
          f"sending {len(todo)}; ~{est:.1f} credits at {args.rate} cr/circuit (cap {args.max_credits})",
          flush=True)
    if args.dry_run or not todo:
        return
    if est > args.max_credits:
        raise SystemExit(f"estimate {est:.1f} > --max-credits {args.max_credits}")

    from dotenv import load_dotenv
    from iqm.qiskit_iqm import transpile_to_IQM

    load_dotenv(args.env)
    if os.environ.get("IQM_PWR_TOKEN"):                          # university account, as in ACIIDS and H2
        os.environ["IQM_TOKEN"] = os.environ["IQM_PWR_TOKEN"]
    os.environ["IQM_COMPUTER"] = "sirius"
    from src.neighborhoods.gate_qaoa import solve as S

    hw = S._get_iqm()
    tc = [transpile_to_IQM(qc, hw, optimization_level=1, seed_transpiler=0) for qc in qcs]
    assert len(tc) <= S.IQM_MAX_CIRCUITS                       # one job
    t0 = time.time()
    job = hw.run(tc, shots=SHOTS)
    print(f"job {job.job_id()} sent", flush=True)
    result = job.result()
    counts = result.get_counts()
    counts = counts if isinstance(counts, list) else [counts]
    S._assert_real_counts(counts, SHOTS, "iqm", len(tc))
    timeline = [{"status": str(e.status), "timestamp": str(e.timestamp)}
                for e in getattr(result, "timeline", [])]                   # QPU time = the bill
    cal = {str(res.calibration_set_id) for res in result.results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as out:
        for r, c, got in zip(todo, tc, counts):
            out.write(json.dumps({"key": r["key"], "day": args.day, "file": r["file"], "inst": r["inst"],
                                  "neigh": r["neigh"], "window": r["window"], "table": r["table"],
                                  "p": r["p"], "K": r["K"], "variables": r["variables"], "shots": SHOTS,
                                  "counts": got, "compiled": _compiled(c, hw), "job_id": job.job_id(),
                                  "calibration_set_id": sorted(cal), "timeline": timeline,
                                  "wall_s": round(time.time() - t0, 1)}) + "\n")
    print(f"{len(tc)} circuits -> {OUT} (calibration set {sorted(cal)}, "
          f"{time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
