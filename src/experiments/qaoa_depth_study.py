"""QAOA depth study (articles/qaoa_depth_icaisc): campaign script.

Subcommand ``e1`` -- window quality versus depth p, noiseless and exact.
The windows are captured from the real gate neighborhoods: while they build a
move, ``windowed.solve_qaoa_batch`` is replaced by a recorder, so every QUBO is
bit for bit the one ILS would send to the device (w = 17 linear, w = 6 dense,
overlap 0.5, no candidate filtering). For each window the optimum is found by
brute force over all 2^K assignments, and for each angle table and p the QAOA
distribution is read exactly from the statevector:

  P_opt       probability of an optimal assignment
  r           <E> / E_min (approximation ratio; windows with E_min = 0 skipped)
  hit_S       1 - (1 - P_opt)^S, chance that the best of S shots is optimal
  best_S      expected QUBO energy of the best of S shots, / E_min
  S95         shots needed to hit an optimum with probability 0.95

p = 0 is the uniform distribution (Hadamards only), the do-nothing baseline.

Angle tables (provenance in each file's "_meta"):
  small  data/qaoa_angles_small.json  w = 6 training windows, p = 1..10
  big    data/qaoa_angles_big.json    w = 17 / w = 6 without the block accelerator, p = 1..5
  ils    data/qaoa_angles_ils.json    windows captured from ILS, p = 1..5

Output: articles/qaoa_depth_icaisc/measurements/e1_window_quality.jsonl, one
line per (window, table, p); reruns skip what is already there.

Subcommand ``e5`` -- ILS on Taillard, noiseless (backend "ideal"), with the
protocol of the Sirius campaign (articles/gate_qaoa/scripts/sirius_full_taillard.py):
ILS, 10 s wall budget, identity start, w = 17 / w = 6 windows, overlap 0.5, PRD
against the Taillard lower bound. Variants:

  A  table small, all four neighborhoods, p = 1..10
  B  best table from E1 for the linear ones: Adjacent "big", Fibonacci "ils", p = 1..5
  C  p = 0 (uniform sampling with energy readout), all four
  D  classical exact neighborhood (same ILS, no QAOA), all four

A-C at 256 and 4096 shots. Linear neighborhoods 1 run, dense 5 runs. Each run is
seeded (ILS kicks and shot sampling), so a rerun of a key reproduces it up to
the number of moves that fit in the wall budget. Output:
articles/qaoa_depth_icaisc/measurements/e5_ils_ideal.jsonl, one line per run.

Usage:
    PYTHONPATH=. .venv311/bin/python3 -m src.experiments.qaoa_depth_study e1 [--workers 4]
    PYTHONPATH=. .venv311/bin/python3 -m src.experiments.qaoa_depth_study e5 [--workers 8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_E1 = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "e1_window_quality.jsonl"

FILES = ["tai20_5", "tai20_10", "tai20_20"]
INSTANCES = range(10)
N_RANDOM_PERMS = 2
NEIGHBORHOODS = ("adjacent", "fibonacci", "dynasearch", "motzkin")
W_LIN, W_GEST, OVERLAP = 17, 6, 0.5
TABLES = {"small": "qaoa_angles_small.json", "big": "qaoa_angles_big.json",
          "ils": "qaoa_angles_ils.json"}
SHOTS = (256, 4096)
# Instances behind the angle calibration (qaoa_calibrate_angles.DEFAULT_INSTANCES).
CALIBRATION = {("tai20_5", 0), ("tai20_5", 1), ("tai20_10", 0)}


def capture_windows(nb: str, pi, pt):
    """The window QUBOs one ILS move of neighborhood ``nb`` would submit."""
    from src.neighborhoods.gate_qaoa import windowed
    from src.neighborhoods.gate_qaoa.adjacent import gate_adjacent_neighborhood
    from src.neighborhoods.gate_qaoa.dynasearch import gate_dynasearch_neighborhood
    from src.neighborhoods.gate_qaoa.fibonacci import gate_fibonacci_neighborhood
    from src.neighborhoods.gate_qaoa.motzkin import gate_motzkin_neighborhood

    fn = {"adjacent": gate_adjacent_neighborhood, "fibonacci": gate_fibonacci_neighborhood,
          "dynasearch": gate_dynasearch_neighborhood, "motzkin": gate_motzkin_neighborhood}[nb]
    w = W_LIN if nb in ("adjacent", "fibonacci") else W_GEST
    captured = []

    def recorder(Qs, neighborhood, **kwargs):
        captured.extend(Qs)
        return [{} for _ in Qs]   # select nothing; the move itself is discarded

    real = windowed.solve_qaoa_batch
    windowed.solve_qaoa_batch = recorder
    try:
        fn(list(pi), pt, p=1, backend="ideal", window_size=w, overlap_ratio=OVERLAP)
    finally:
        windowed.solve_qaoa_batch = real
    return [Q for Q in captured if Q]


def energies(Q, variables):
    """QUBO energy of every assignment, indexed like Statevector.probabilities()
    (bit k of the index = qubit k = variables[k])."""
    from src.neighborhoods.gate_qaoa.circuit import qubo_energy

    K = len(variables)
    idx = np.arange(2 ** K)
    x = ((idx[:, None] >> np.arange(K)) & 1).astype(float)
    pos = {v: k for k, v in enumerate(variables)}
    E = np.zeros(2 ** K)
    for (a, b), c in Q.items():
        E += c * (x[:, pos[a]] if a == b else x[:, pos[a]] * x[:, pos[b]])
    for i in (0, 2 ** K - 1, int(idx[len(idx) // 3])):   # spot check against the reference
        assign = {v: int((i >> k) & 1) for k, v in enumerate(variables)}
        assert abs(E[i] - qubo_energy(Q, assign)) < 1e-6
    return E


def metrics(prob, E):
    e_min = float(E.min())
    opt = np.abs(E - e_min) < 1e-9
    p_opt = float(prob[opt].sum())
    out = {"P_opt": p_opt, "mean_E": float(prob @ E)}
    out["r"] = out["mean_E"] / e_min if e_min < 0 else None
    order = np.argsort(E, kind="stable")
    levels, start = np.unique(E[order], return_index=True)
    mass = np.add.reduceat(prob[order], start)          # probability of each energy level
    cdf = np.cumsum(mass)
    for S in SHOTS:
        out[f"hit_{S}"] = 1.0 - (1.0 - p_opt) ** S
        p_min = (1.0 - np.concatenate([[0.0], cdf[:-1]])) ** S - (1.0 - cdf) ** S
        best = float(levels @ p_min)
        out[f"best_{S}"] = best / e_min if e_min < 0 else None
    out["S95"] = (math.log(0.05) / math.log(1.0 - p_opt)) if 0 < p_opt < 1 else (1.0 if p_opt >= 1 else None)
    return out


def _angles(table: str, nb: str):
    f = REPO / "data" / TABLES[table]
    return json.loads(f.read_text()).get(nb) if f.exists() else None


def e1_task(args):
    """All windows, tables and depths of one (file, instance)."""
    fname, inst, done = args
    from qiskit.quantum_info import Statevector
    from src.neighborhoods.gate_qaoa.circuit import build_qaoa_circuit, normalize_ising, qubo_to_ising
    from src.parser import parser

    data = parser(str(REPO / "data" / f"{fname}.txt"), inst)
    n, pt = data["info"]["jobs"], data["processing_times"]
    rng = np.random.default_rng(1000 * FILES.index(fname) + inst)
    perms = [("identity", list(range(n)))] + [
        (f"random{k}", [int(v) for v in rng.permutation(n)]) for k in range(N_RANDOM_PERMS)]
    rows = []
    for perm_name, pi in perms:
        for nb in NEIGHBORHOODS:
            tables = {t: _angles(t, nb) for t in TABLES}
            for wi, Q in enumerate(capture_windows(nb, pi, pt)):
                variables, h, J, _ = qubo_to_ising(Q)
                K = len(variables)
                E = energies(Q, variables)
                hn, Jn, _ = normalize_ising(h, J)
                base = {"file": fname, "inst": inst, "perm": perm_name, "neigh": nb, "window": wi,
                        "K": K, "E_min": float(E.min()),
                        "n_opt": int((np.abs(E - E.min()) < 1e-9).sum()),
                        "calibration_instance": (fname, inst) in CALIBRATION}
                todo = [("uniform", 0, None)]
                for t, tab in tables.items():
                    if tab:
                        todo += [(t, int(p), tab[p]) for p in sorted(tab, key=int)]
                for t, p, ang in todo:
                    key = f"{fname}|{inst}|{perm_name}|{nb}|{wi}|{t}|{p}"
                    if key in done:
                        continue
                    if p == 0:
                        prob = np.full(2 ** K, 2.0 ** -K)
                    else:
                        qc = build_qaoa_circuit(K, hn, Jn, ang["gamma"], ang["beta"])
                        prob = Statevector(qc).probabilities()
                    rows.append({"key": key, **base, "table": t, "p": p, **metrics(prob, E)})
    return rows


def run_e1(workers: int):
    OUT_E1.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_E1.exists():
        done = {json.loads(line)["key"] for line in OUT_E1.read_text().splitlines() if line}
    missing = [f"{t}:{nb}" for t in TABLES for nb in NEIGHBORHOODS if _angles(t, nb) is None]
    if missing:
        print("angle tables not ready yet (rerun later to fill in):", ", ".join(missing))
    tasks = [(f, i, done) for f in FILES for i in INSTANCES]
    with Pool(workers) as pool, OUT_E1.open("a") as out:
        for k, rows in enumerate(pool.imap_unordered(e1_task, tasks), 1):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"[{k}/{len(tasks)}] +{len(rows)} rows", flush=True)


OUT_E5 = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "e5_ils_ideal.jsonl"
BUDGET_MS = 10_000
BUDGETS = [100, 500, 1000, 2000, 5000, 10000]
REPS = {"adjacent": 1, "fibonacci": 1, "dynasearch": 5, "motzkin": 5}


def budget_columns(move_log, budgets=BUDGETS):
    """Best Cmax for each budget, replaying the ILS stop rule: move k counts when
    its START (the end of move k-1) was before the budget (as in the Sirius campaign)."""
    out = {}
    for b in budgets:
        k = 0
        while k < len(move_log) and (0 if k == 0 else move_log[k - 1][0]) < b:
            k += 1
        out[str(b)] = move_log[k - 1][1] if k else None
    return out


def e5_tasks():
    """(variant, file, inst, neigh, table, p, shots, rep), runs of one rep level first."""
    tasks = []
    for rep in range(max(REPS.values())):
        for f in FILES:
            for i in INSTANCES:
                for nb in NEIGHBORHOODS:
                    if rep >= REPS[nb]:
                        continue
                    for shots in SHOTS:
                        tasks += [("A", f, i, nb, "small", p, shots, rep) for p in range(1, 11)]
                        if nb == "adjacent":
                            tasks += [("B", f, i, nb, "big", p, shots, rep) for p in range(1, 6)]
                        if nb == "fibonacci":
                            tasks += [("B", f, i, nb, "ils", p, shots, rep) for p in range(1, 6)]
                        tasks.append(("C", f, i, nb, "uniform", 0, shots, rep))
                    tasks.append(("D", f, i, nb, "classical", None, None, rep))
    return tasks


def e5_key(task):
    v, f, i, nb, table, p, shots, rep = task
    return f"{f}|{i}|{nb}|{table}|{p}|{shots}|{rep}"


def e5_task(args):
    task, budget_ms = args
    v, f, i, nb, table, p, shots, rep = task
    import random
    import time
    from src.algorithms.ils import iterated_local_search
    from src.neighborhoods.gate_qaoa import solve as S
    from src.parser import parser
    from src.permutation_procesing import c_max

    key = e5_key(task)
    seed = abs(hash(key)) % (2 ** 31)   # PYTHONHASHSEED is fixed below for stable seeds
    random.seed(seed)
    S.set_sampler_seed(seed)
    if table in TABLES:
        path = REPO / "data" / TABLES[table]
        if S._ANGLES_PATH != path:
            S._ANGLES_PATH, S._ANGLES_CACHE = path, None
    inst = parser(str(REPO / "data" / f"{f}.txt"), i)
    pt, lb = inst["processing_times"], inst["info"]["lower_bound"]
    qc = None
    mode = nb
    if v != "D":
        mode = f"gate_{nb}"
        qc = {"qaoa_backend": "ideal", "qaoa_p": p, "qaoa_shots": shots,
              "qaoa_window_size_linear": W_LIN, "qaoa_window_size": W_GEST,
              "qaoa_overlap_ratio": OVERLAP}
    t0 = time.time()
    _, cmax, _, _, st = iterated_local_search(pt, max_time_ms=budget_ms, tabu_tenure=10,
                                              neigh_mode=mode, quantum_config=qc)
    wall = time.time() - t0
    ml = [list(x) for x in st["move_log"]]
    return {"key": key, "variant": v, "file": f, "inst": i, "neigh": nb, "table": table,
            "p": p, "shots": shots, "rep": rep, "seed": seed, "lb": lb,
            "cmax_start": c_max(list(range(len(pt[0]))), pt), "cmax": cmax,
            "prd": 100.0 * (cmax - lb) / lb, "moves": st["iterations"],
            "neigh_time_ms": st["neigh_time_ms"], "wall_s": wall,
            "ms_per_move": st["neigh_time_ms"] / max(1, st["iterations"]),
            "move_log": ml, "budget_columns": budget_columns(ml)}


def run_e5(workers: int, smoke: bool):
    import os

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("run e5 with PYTHONHASHSEED=0 (stable per-run seeds).")
    out_path = OUT_E5.with_name("e5_smoke.jsonl") if smoke else OUT_E5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line}
    tasks = e5_tasks()
    budget = BUDGET_MS
    if smoke:   # one instance, a few configurations of every variant, 1 s budget
        pick = {("A", 1), ("A", 10), ("B", 3), ("C", 0), ("D", None)}
        tasks = [t for t in tasks if t[1] == "tai20_5" and t[2] == 0 and t[7] == 0
                 and (t[0], t[5]) in pick and t[6] in (4096, None)]
        budget = 1000
    todo = [(t, budget) for t in tasks if e5_key(t) not in done]
    print(f"{len(tasks)} runs planned, {len(tasks) - len(todo)} done, {len(todo)} to go, "
          f"budget {budget} ms, {workers} workers", flush=True)
    with Pool(workers) as pool, out_path.open("a") as out:
        for k, row in enumerate(pool.imap_unordered(e5_task, todo), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if k % 20 == 0 or smoke:
                print(f"[{k}/{len(todo)}] {row['key']}: PRD {row['prd']:.2f}% "
                      f"moves {row['moves']} ({row['ms_per_move']:.0f} ms/move)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e1 = sub.add_parser("e1", help="window quality vs p (noiseless, exact)")
    e1.add_argument("--workers", type=int, default=4)
    e5 = sub.add_parser("e5", help="ILS on Taillard, noiseless, all variants")
    e5.add_argument("--workers", type=int, default=8)
    e5.add_argument("--smoke", action="store_true", help="tiny test into e5_smoke.jsonl")
    args = ap.parse_args()
    if args.cmd == "e1":
        run_e1(args.workers)
    elif args.cmd == "e5":
        run_e5(args.workers, args.smoke)


if __name__ == "__main__":
    main()
