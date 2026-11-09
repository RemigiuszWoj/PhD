"""T5: ILS on Taillard with a simulator in place of Sirius -> PRD per instance.

Every Sirius p = 1 run of the ACIIDS campaign
(articles/gate_qaoa/measurements/sirius_full_taillard_ils.json) is replayed
with the protocol of that campaign (ILS, identity start, tabu 10, w = 17
linear / w = 6 dense, overlap 0.5, 4096 shots) and with as many moves as
Sirius made in its 10 s, as E5n of the ICAISC study does. The QAOA circuits of
every move go through backend.SimDevice: compiled by qiskit-iqm for the star,
executed by one engine under S1/S2/S3, read out with the snapshot's readout
errors. The result is the PRD of each run, next to the PRD Sirius reached.

Angles: p = 1 uses the default table data/qaoa_angles.json, as the Sirius runs
did; p >= 2 the H2 tables (adjacent "big", the others "small").
Engines: trajectory methods only (qsim, qulacs, qulacs_loop, aer_sv, qibo);
a move has windows up to K = 15 (16 qubits). aer_sv fails in S3 and qibo and
qulacs (NoiseSimulator) are wrong or unsupported there (see stage 2).

Usage (from .venv-annsim; each worker process opens its own compile worker in .venv-iqm):
    PYTHONHASHSEED=0 PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.t5_ils \\
        --engines qsim qulacs_loop --scenarios S2 [--files tai20_5 ...] [--p 1] [--jobs 8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from .noise import REPO

SIRIUS_ILS = REPO / "articles" / "gate_qaoa" / "measurements" / "sirius_full_taillard_ils.json"
OUT = REPO / "articles" / "annsim_simulators" / "measurements" / "t5_ils.jsonl"
W_LIN, W_GEST, OVERLAP = 17, 6, 0.5
H2_TABLE = {"adjacent": "qaoa_angles_big.json", "fibonacci": "qaoa_angles_small.json",
            "dynasearch": "qaoa_angles_small.json", "motzkin": "qaoa_angles_small.json"}
_DEVICES = {}


def sirius_runs(files):
    s = json.loads(SIRIUS_ILS.read_text())
    return [v for k, v in s.items() if not k.startswith("_") and (not files or v["plik"] in files)]


def task_key(v, engine, scenario, p):
    return f"{v['plik']}|{v['inst']}|{v['neigh'].replace('gate_', '')}|{v['powt']}|{engine}|{scenario}|{p}"


def _device(engine, scenario):
    """One SimDevice per (engine, scenario) and process; all share the process's compile worker."""
    from .backend import CompileWorker, SimDevice

    if (engine, scenario) not in _DEVICES:
        worker = next(iter(_DEVICES.values())).worker if _DEVICES else CompileWorker()
        _DEVICES[(engine, scenario)] = SimDevice(engine, scenario, worker)
    return _DEVICES[(engine, scenario)]


def t5_task(args):
    v, engine, scenario, p = args
    import random

    from src.algorithms.ils import iterated_local_search
    from src.neighborhoods.gate_qaoa import solve as S
    from src.parser import parser

    from .backend import install

    key = task_key(v, engine, scenario, p)
    nb = v["neigh"].replace("gate_", "")
    seed = abs(hash(key)) % (2 ** 31)              # PYTHONHASHSEED=0 makes it stable
    random.seed(seed)
    S.set_sampler_seed(seed)
    table = REPO / "data" / ("qaoa_angles.json" if p == 1 else H2_TABLE[nb])
    if S._ANGLES_PATH != table:
        S._ANGLES_PATH, S._ANGLES_CACHE = table, None
    dev = _device(engine, scenario)
    before = dict(dev.stats)
    undo = install(dev)
    inst = parser(str(REPO / "data" / f"{v['plik']}.txt"), v["inst"])
    pt, lb = inst["processing_times"], inst["info"]["lower_bound"]
    qc = {"qaoa_backend": "ideal", "qaoa_p": p, "qaoa_shots": 4096,       # "ideal" never reaches a device:
          "qaoa_window_size_linear": W_LIN, "qaoa_window_size": W_GEST,   # install() replaced the call
          "qaoa_overlap_ratio": OVERLAP}
    t0 = time.time()
    try:
        _, cmax, _, _, st = iterated_local_search(pt, max_time_ms=10 ** 9, tabu_tenure=10,
                                                  neigh_mode=f"gate_{nb}", quantum_config=qc,
                                                  max_moves=v["ruchow"])
    except Exception as exc:
        return {"key": key, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    finally:
        undo()
    return {"key": key, "ok": True, "file": v["plik"], "inst": v["inst"], "neigh": nb, "rep": v["powt"],
            "engine": engine, "scenario": scenario, "p": p, "table": table.name, "seed": seed,
            "moves_budget": v["ruchow"], "moves": st["iterations"], "lb": lb, "cmax": cmax,
            "prd": 100.0 * (cmax - lb) / lb, "prd_sirius": v["prd"], "cmax_sirius": v["cmax"],
            "move_log": [list(x) for x in st["move_log"]], "wall_s": time.time() - t0,
            **{k: dev.stats[k] - before[k] for k in dev.stats}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", nargs="+", default=["qsim", "qulacs_loop"])
    ap.add_argument("--scenarios", nargs="+", default=["S2"])
    ap.add_argument("--files", nargs="*", default=["tai20_5", "tai20_10", "tai20_20"])
    ap.add_argument("--p", nargs="+", type=int, default=[1])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true", help="one run of tai20_5 inst 0 per engine")
    args = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run with PYTHONHASHSEED=0 (stable per-run seeds).")
    runs = sirius_runs(set(args.files))
    if args.smoke:
        runs = [v for v in runs if v["plik"] == "tai20_5" and v["inst"] == 0 and v["powt"] == 0][:1]
    done = set()
    if args.out.exists():
        done = {r["key"] for r in map(json.loads, args.out.read_text().splitlines()) if r.get("ok")}
    tasks = [(v, e, s, p) for p in args.p for s in args.scenarios for e in args.engines for v in runs
             if task_key(v, e, s, p) not in done]
    tasks.sort(key=lambda t: int(t[0]["plik"][3:].split("_")[0]), reverse=True)   # big instances first
    print(f"{len(runs)} Sirius runs, {len(tasks)} ILS runs to go, {args.jobs} at a time", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.jobs) as pool, args.out.open("a") as fh:
        for k, row in enumerate(pool.imap_unordered(t5_task, tasks), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            msg = (f"PRD {row['prd']:.2f}% (Sirius {row['prd_sirius']:.2f}%), {row['moves']} moves, "
                   f"{row['circuits']} circuits, {row['wall_s']:.0f} s") if row.get("ok") else row["error"]
            print(f"[{k}/{len(tasks)}] {row['key']}: {msg}", flush=True)


if __name__ == "__main__":
    main()
