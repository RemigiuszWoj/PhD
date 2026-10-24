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

Subcommands on the noisy side (run from .venv-iqm, which has iqm-client and Aer):

  e3   circuit cost versus p = 1..10: the E4 windows compiled to the fake Sirius
       (star) and to IQMFakeApollo (square lattice); 2q/1q gate counts and depths.
  e4   window quality under noise: the E1 windows of instances 0-3 (identity
       permutation), best E1 table per neighborhood, p = 0..6, fake Sirius
       "calibrated", 4096 shots; "thermal" on instance 0 at p = 1, 3, 6 as an
       upper bound on the noise. Readout as in ILS (lowest-energy sample).
  e5n  ILS under noise with a MOVE budget: every Sirius p = 1 run of the ACIIDS
       campaign (n = 20, instances 0-3) is replayed with as many moves as Sirius
       made in its 10 s, p = 0..6, fake Sirius "calibrated", 4096 shots.

Usage:
    PYTHONPATH=. .venv311/bin/python3 -m src.experiments.qaoa_depth_study e1 [--workers 4]
    PYTHONPATH=. .venv311/bin/python3 -m src.experiments.qaoa_depth_study e5 [--workers 8] [--smoke]
    PYTHONHASHSEED=0 PYTHONPATH=. .venv-iqm/bin/python -m src.experiments.qaoa_depth_study {e3|e4|e5n} [--workers 8] [--smoke]
    PYTHONHASHSEED=0 PYTHONPATH=. .venv-iqm/bin/python -m src.experiments.qaoa_depth_study h2 [--dry-run] [--max-credits 230]

Subcommand ``h2`` -- ILS on the real IQM Sirius at p = 2 and 3 with the ACIIDS
protocol (ILS, 10 s, 4096 shots); p = 1 comes from the ACIIDS campaign. It
replays run 0 of every Sirius p = 1 run (tai20, instances 0-3, 48 runs), all
p = 2 runs first, so a cut-off budget leaves complete p = 2. Circuits are
counted as they are submitted; the run stops before the estimated spend
(circuits x --rate) would pass --max-credits. IQM reports no credit balance
through the API, so check resonance.iqm.tech/jobs after the first runs. Run
it only while nothing else loads the CPU: the budget is wall time. Take a
fresh calibration snapshot first (src/experiments/iqm_sirius_snapshot.py).
"""
from __future__ import annotations

import argparse
import collections
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


# ---------------------------------------------------------------- noisy side
OUT_E3 = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "e3_circuit_cost.jsonl"
OUT_E4 = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "e4_window_noise.jsonl"
OUT_E5N = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "e5n_ils_noisy.jsonl"
SIRIUS_ILS = REPO / "articles" / "gate_qaoa" / "measurements" / "sirius_full_taillard_ils.json"
NOISY_INSTANCES = range(4)            # the instances Sirius ran at p = 1 for every n = 20 file
NOISY_P = range(0, 7)
# Best angle table per neighborhood from E1 (WYNIKI.md, W5); Fibonacci has no stable winner.
# "big" and "ils" stop at p = 5, so Adjacent also runs "small" to cover p = 6.
BEST_TABLES = {"adjacent": ["big", "small"], "fibonacci": ["small", "ils"],
               "dynasearch": ["small"], "motzkin": ["small"]}


def _depths(table: str, nb: str):
    """p > 0 of NOISY_P that the angle table actually has."""
    return [p for p in NOISY_P if p > 0 and str(p) in _angles(table, nb)]
_FAKES = {}


def _fake(kind: str):
    """Cached fake backend per worker: sirius_calibrated | sirius_thermal | apollo."""
    if kind not in _FAKES:
        if kind == "apollo":
            from iqm.qiskit_iqm.fake_backends import IQMFakeApollo
            _FAKES[kind] = IQMFakeApollo()
        else:
            from src.neighborhoods.gate_qaoa.fake_sirius import fake_sirius
            _FAKES[kind] = fake_sirius(mode=kind.split("_", 1)[1])
    return _FAKES[kind]


def _circuit(Q, table, nb, p):
    """Measured QAOA circuit of window Q exactly as solve.py builds it."""
    from src.neighborhoods.gate_qaoa.circuit import build_qaoa_circuit, normalize_ising, qubo_to_ising

    variables, h, J, _ = qubo_to_ising(Q)
    hn, Jn, _ = normalize_ising(h, J)
    ang = {"gamma": [], "beta": []} if p == 0 else _angles(table, nb)[str(p)]
    return variables, build_qaoa_circuit(len(variables), hn, Jn, ang["gamma"], ang["beta"], measure=True)


def _noisy_windows():
    """(file, inst, neigh, window index, Q) for the E1 windows of the noisy study."""
    from src.parser import parser

    out = []
    for f in FILES:
        for i in NOISY_INSTANCES:
            data = parser(str(REPO / "data" / f"{f}.txt"), i)
            n, pt = data["info"]["jobs"], data["processing_times"]
            for nb in NEIGHBORHOODS:
                for wi, Q in enumerate(capture_windows(nb, list(range(n)), pt)):
                    out.append((f, i, nb, wi, Q))
    return out


def _run_pool(out_path, todo, worker, workers, label):
    print(f"{label}: {len(todo)} to go, {workers} workers -> {out_path.name}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool, out_path.open("a") as out:
        for k, row in enumerate(pool.imap_unordered(worker, todo), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[{k}/{len(todo)}] {row['key']} ({row.get('sim_s', row.get('wall_s', 0)):.0f} s)", flush=True)


def _done(path):
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line}


def e3_task(args):
    f, i, nb, wi, Q = args
    from iqm.qiskit_iqm import transpile_to_IQM

    rows = []
    for kind in ("sirius_calibrated", "apollo"):
        for p in range(1, 11):
            _, qc = _circuit(Q, "small", nb, p)   # gate counts barely depend on the angles
            tc = transpile_to_IQM(qc, _fake(kind), optimization_level=1)
            ops = dict(tc.count_ops())
            rows.append({"topology": "star" if kind.startswith("sirius") else "lattice", "p": p,
                         "cz": ops.get("cz", 0), "move": ops.get("move", 0), "r": ops.get("r", 0),
                         "depth": tc.depth(),
                         "depth_2q": tc.depth(filter_function=lambda x: x.operation.num_qubits == 2)})
    return {"key": f"{f}|{i}|{nb}|{wi}", "file": f, "inst": i, "neigh": nb, "window": wi,
            "K": len(_circuit(Q, "small", nb, 1)[0]), "table": "small",
            "cost": rows}


def e4_task(args):
    (f, i, nb, wi, Q), table, p, mode = args
    import time
    from iqm.qiskit_iqm import transpile_to_IQM

    key = f"{f}|{i}|{nb}|{wi}|{table}|{p}|{mode}"
    seed = abs(hash(key)) % (2 ** 31)
    variables, qc = _circuit(Q, table, nb, p)
    E = energies(Q, variables)
    e_min = float(E.min())
    fake = _fake(f"sirius_{mode}")
    t0 = time.time()
    tc = transpile_to_IQM(qc, fake, optimization_level=1)
    got = fake.run(tc, shots=4096, seed_simulator=seed).result().get_counts()
    sim_s = time.time() - t0
    counts = got[0] if isinstance(got, list) else got
    shots = sum(counts.values())
    if shots < 2048:                     # same guard as _assert_real_counts
        raise RuntimeError(f"{key}: {shots} shots came back")
    idx = {bs: int(bs.replace(" ", ""), 2) for bs in counts}       # bit k = qubit k, as in E1
    opt = np.abs(E - e_min) < 1e-9
    p_opt = sum(c for bs, c in counts.items() if opt[idx[bs]]) / shots
    pick = min(counts, key=lambda bs: E[idx[bs]])
    e_pick = float(E[idx[pick]])
    return {"key": key, "file": f, "inst": i, "neigh": nb, "window": wi, "table": table,
            "p": p, "mode": mode, "K": len(variables), "E_min": e_min, "seed": seed,
            "P_opt": p_opt, "mean_E": sum(E[idx[bs]] * c for bs, c in counts.items()) / shots,
            "pick_E": e_pick, "pick_ratio": e_pick / e_min if e_min < 0 else None,
            "pick_is_opt": bool(opt[idx[pick]]), "distinct": len(counts),
            "cz": dict(tc.count_ops()).get("cz", 0), "move": dict(tc.count_ops()).get("move", 0),
            "depth": tc.depth(), "sim_s": sim_s}


def run_e3(workers: int, smoke: bool):
    wins = _noisy_windows()
    out = OUT_E3.with_name("e3_smoke.jsonl") if smoke else OUT_E3
    done = _done(out)
    todo = [w for w in wins if f"{w[0]}|{w[1]}|{w[2]}|{w[3]}" not in done]
    if smoke:
        todo = [w for w in todo if w[0] == "tai20_5" and w[1] == 0 and w[3] == 0]
    _run_pool(out, todo, e3_task, workers, "e3")


def run_e4(workers: int, smoke: bool):
    wins = [w for w in _noisy_windows() if (lambda E: E.min() < 0)(energies(w[4], _circuit(w[4], BEST_TABLES[w[2]][0], w[2], 0)[0]))]
    tasks = []
    for w in wins:
        nb = w[2]
        tasks.append((w, "uniform", 0, "calibrated"))
        for table in BEST_TABLES[nb]:
            tasks += [(w, table, p, "calibrated") for p in _depths(table, nb)]
            if w[1] == 0 and table == BEST_TABLES[nb][0]:
                tasks += [(w, table, p, "thermal") for p in (1, 3, 6)]
    out = OUT_E4.with_name("e4_smoke.jsonl") if smoke else OUT_E4
    done = _done(out)
    key = lambda t: f"{t[0][0]}|{t[0][1]}|{t[0][2]}|{t[0][3]}|{t[1]}|{t[2]}|{t[3]}"
    todo = [t for t in tasks if key(t) not in done]
    if smoke:
        todo = [t for t in todo if t[0][0] == "tai20_5" and t[0][1] == 0 and t[0][3] == 0
                and t[2] in (0, 1)]
    # Longest first (dense, high p), so the pool does not end on one slow straggler.
    todo.sort(key=lambda t: (t[0][2] in ("dynasearch", "motzkin"), t[2]), reverse=True)
    _run_pool(out, todo, e4_task, workers, f"e4 ({len(wins)} windows with an improving move)")


def e5n_task(args):
    (f, i, nb, rep, moves), table, p = args
    import os
    import random
    import time
    from src.algorithms.ils import iterated_local_search
    from src.neighborhoods.gate_qaoa import solve as S
    from src.parser import parser

    os.environ["QAOA_NOISE_BACKEND"] = "sirius"
    key = f"{f}|{i}|{nb}|{table}|{p}|{rep}"
    seed = abs(hash(key)) % (2 ** 31)
    random.seed(seed)
    S.set_sampler_seed(seed)
    if table in TABLES:
        path = REPO / "data" / TABLES[table]
        if S._ANGLES_PATH != path:
            S._ANGLES_PATH, S._ANGLES_CACHE = path, None
    inst = parser(str(REPO / "data" / f"{f}.txt"), i)
    pt, lb = inst["processing_times"], inst["info"]["lower_bound"]
    qc = {"qaoa_backend": "aer_noisy", "qaoa_p": p, "qaoa_shots": 4096,
          "qaoa_window_size_linear": W_LIN, "qaoa_window_size": W_GEST, "qaoa_overlap_ratio": OVERLAP}
    t0 = time.time()
    _, cmax, _, _, st = iterated_local_search(pt, max_time_ms=10 ** 9, tabu_tenure=10,
                                              neigh_mode=f"gate_{nb}", quantum_config=qc,
                                              max_moves=moves)
    return {"key": key, "file": f, "inst": i, "neigh": nb, "table": table, "p": p, "rep": rep,
            "seed": seed, "moves_budget": moves, "moves": st["iterations"], "lb": lb, "cmax": cmax,
            "prd": 100.0 * (cmax - lb) / lb, "move_log": [list(x) for x in st["move_log"]],
            "wall_s": time.time() - t0}


def run_e5n(workers: int, smoke: bool):
    sirius = json.loads(SIRIUS_ILS.read_text())
    runs = []                                   # (file, inst, neigh, rep, moves Sirius made in 10 s)
    for k, v in sirius.items():
        if k.startswith("_") or v["plik"] not in FILES or v["inst"] not in NOISY_INSTANCES:
            continue
        runs.append((v["plik"], v["inst"], v["neigh"].replace("gate_", ""), v["powt"], v["ruchow"]))
    tasks = []
    for r in runs:
        tasks.append((r, "uniform", 0))
        for table in BEST_TABLES[r[2]]:
            tasks += [(r, table, p) for p in _depths(table, r[2])]
    out = OUT_E5N.with_name("e5n_smoke.jsonl") if smoke else OUT_E5N
    done = _done(out)
    key = lambda t: f"{t[0][0]}|{t[0][1]}|{t[0][2]}|{t[1]}|{t[2]}|{t[0][3]}"
    todo = [t for t in tasks if key(t) not in done]
    if smoke:
        todo = [t for t in todo if t[0][0] == "tai20_5" and t[0][1] == 0 and t[0][3] == 0 and t[2] == 1]
    todo.sort(key=lambda t: (t[0][2] in ("dynasearch", "motzkin"), t[2]), reverse=True)
    print(f"{len(runs)} Sirius p=1 runs replayed; moves per run: "
          f"{sorted({(r[2], r[4]) for r in runs})}", flush=True)
    _run_pool(out, todo, e5n_task, workers, "e5n")


# ----------------------------------------------------------------- hardware
OUT_H2 = REPO / "articles" / "qaoa_depth_icaisc" / "measurements" / "h2_sirius_ils.jsonl"
H2_P = (2, 3)
H2_TABLE = {"adjacent": "big", "fibonacci": "small", "dynasearch": "small", "motzkin": "small"}


def _h2_runs():
    """Run 0 of every Sirius p = 1 run on tai20, instances 0-3: (file, inst, neigh, moves)."""
    sirius = json.loads(SIRIUS_ILS.read_text())
    return [(v["plik"], v["inst"], v["neigh"].replace("gate_", ""), v["ruchow"])
            for k, v in sirius.items()
            if not k.startswith("_") and v["plik"] in FILES and v["inst"] in NOISY_INSTANCES
            and v["powt"] == 0]


def _h2_windows():
    """Number of windows of the first ILS move per (file, inst, neigh), from E1."""
    w = collections.Counter()
    for line in OUT_E1.read_text().splitlines():
        r = json.loads(line)
        if r["table"] == "uniform" and r["perm"] == "identity" and r["inst"] in NOISY_INSTANCES:
            w[(r["file"], r["inst"], r["neigh"])] += 1
    return w


def run_h2(dry_run: bool, max_credits: float, rate: float):
    import os
    import random
    import time

    runs = _h2_runs()
    tasks = [(r, p) for p in H2_P for r in runs]                 # all p = 2 first
    done = _done(OUT_H2)
    key = lambda r, p: f"{r[0]}|{r[1]}|{r[2]}|{H2_TABLE[r[2]]}|{p}|0"
    todo = [(r, p) for r, p in tasks if key(r, p) not in done]
    windows = _h2_windows()
    est = {key(r, p): r[3] * windows[(r[0], r[1], r[2])] for r, p in todo}
    total = sum(est.values())
    print(f"h2: {len(tasks)} runs planned, {len(tasks) - len(todo)} done, {len(todo)} to go; "
          f"~{total} circuits = ~{total * rate:.0f} credits at {rate} cr/circuit "
          f"(cap {max_credits})", flush=True)
    if dry_run:
        by = collections.Counter()
        for (r, p), (k, c) in zip(todo, est.items()):
            by[(p, r[2])] += c
        for (p, nb), c in sorted(by.items()):
            print(f"  p={p} {nb:10s} ~{c:4d} circuits ~{c * rate:6.1f} cr")
        return

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    if os.environ.get("IQM_PWR_TOKEN"):                          # university account, as in ACIIDS
        os.environ["IQM_TOKEN"] = os.environ["IQM_PWR_TOKEN"]
    os.environ["IQM_COMPUTER"] = "sirius"
    from src.algorithms.ils import iterated_local_search
    from src.neighborhoods.gate_qaoa import solve as S
    from src.parser import parser

    submitted = [0]
    real_batch = S._device_counts_batch

    def counting(circuits, backend, shots):
        if (submitted[0] + len(circuits)) * rate > max_credits:
            raise RuntimeError(f"credit cap {max_credits} reached ({submitted[0]} circuits sent)")
        out = real_batch(circuits, backend, shots)
        submitted[0] += len(circuits)
        return out

    S._device_counts_batch = counting
    OUT_H2.parent.mkdir(parents=True, exist_ok=True)
    with OUT_H2.open("a") as out:
        for r, p in todo:
            f, i, nb, moves_p1 = r
            k = key(r, p)
            seed = abs(hash(k)) % (2 ** 31)
            random.seed(seed)
            table = H2_TABLE[nb]
            S._ANGLES_PATH, S._ANGLES_CACHE = REPO / "data" / TABLES[table], None
            inst = parser(str(REPO / "data" / f"{f}.txt"), i)
            pt, lb = inst["processing_times"], inst["info"]["lower_bound"]
            qc = {"qaoa_backend": "iqm", "qaoa_p": p, "qaoa_shots": 4096,
                  "qaoa_window_size_linear": W_LIN, "qaoa_window_size": W_GEST,
                  "qaoa_overlap_ratio": OVERLAP}
            before = submitted[0]
            t0 = time.time()
            try:
                _, cmax, _, _, st = iterated_local_search(pt, max_time_ms=BUDGET_MS, tabu_tenure=10,
                                                          neigh_mode=f"gate_{nb}", quantum_config=qc)
            except RuntimeError as e:
                print(f"STOP at {k}: {e}", flush=True)
                break
            ml = [list(x) for x in st["move_log"]]
            row = {"key": k, "file": f, "inst": i, "neigh": nb, "table": table, "p": p, "rep": 0,
                   "seed": seed, "lb": lb, "cmax": cmax, "prd": 100.0 * (cmax - lb) / lb,
                   "moves": st["iterations"], "moves_p1_sirius": moves_p1,
                   "circuits": submitted[0] - before, "wall_s": time.time() - t0,
                   "move_log": ml, "budget_columns": budget_columns(ml),
                   "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{k}: PRD {row['prd']:.2f}% moves {row['moves']} circuits {row['circuits']} "
                  f"(total {submitted[0]} ~{submitted[0] * rate:.0f} cr)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e1 = sub.add_parser("e1", help="window quality vs p (noiseless, exact)")
    e1.add_argument("--workers", type=int, default=4)
    e5 = sub.add_parser("e5", help="ILS on Taillard, noiseless, all variants")
    e5.add_argument("--workers", type=int, default=8)
    e5.add_argument("--smoke", action="store_true", help="tiny test into e5_smoke.jsonl")
    for name, help_ in (("e3", "circuit cost vs p, star and lattice"),
                        ("e4", "window quality under noise (fake Sirius)"),
                        ("e5n", "ILS under noise, move budget from Sirius")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--workers", type=int, default=8)
        s.add_argument("--smoke", action="store_true")
    h2 = sub.add_parser("h2", help="ILS on the real Sirius, p = 2 and 3 (uses credits)")
    h2.add_argument("--dry-run", action="store_true", help="count circuits and credits, submit nothing")
    h2.add_argument("--max-credits", type=float, default=230.0)
    h2.add_argument("--rate", type=float, default=0.75, help="credits per circuit (NOTES.md: 0.75)")
    args = ap.parse_args()
    if args.cmd == "h2":
        run_h2(args.dry_run, args.max_credits, args.rate)
        return
    if args.cmd in ("e3", "e4", "e5n"):
        import os
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise SystemExit("run with PYTHONHASHSEED=0 (stable per-run seeds).")
        {"e3": run_e3, "e4": run_e4, "e5n": run_e5n}[args.cmd](args.workers, args.smoke)
        return
    if args.cmd == "e1":
        run_e1(args.workers)
    elif args.cmd == "e5":
        run_e5(args.workers, args.smoke)


if __name__ == "__main__":
    main()
