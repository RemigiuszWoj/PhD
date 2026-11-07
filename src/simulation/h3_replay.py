"""H3 replay: the compiled circuits that ran on Sirius, on the simulators, with the same-day calibration.

Every row of h3_sirius.jsonl keeps the compiled gate list that ran on the
device (identical on both days). Here each of them runs on every
(engine, scenario) with the noise of the calibration snapshot of its own day,
4096 shots, read out with that snapshot's readout errors as in backend.py, so
the simulated histogram and the device histogram are sampled the same way.
analyse() then compares, per circuit, simulator vs device with device vs
device between the two days: the total variation distance (TVD; with 4096
shots over up to 2^15 outcomes it is mostly sampling noise), the mean QUBO
energy, the share of shots on the optimum and the mean Hamming weight.

Usage (from the annsim-simulators worktree, .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.h3_replay [--jobs 8] [--smoke]
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.h3_replay --analyse
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .h3 import OUT as H3, SHOTS
from .noise import REPO

OUT = H3.with_name("h3_replay.jsonl")
SNAPSHOT = {"2026-11-06": "iqm_sirius_calibration_20261106.json",
            "2026-11-07": "iqm_sirius_calibration_20261107.json"}
ENGINES = ("qsim", "qulacs_loop")
SCENARIOS = ("S1", "S2", "S3")


def task_key(t):
    return f"{t['key']}|{t['day']}|{t['engine']}|{t['scenario']}"


def _simulate(t):
    """One compiled circuit on one engine and scenario; runs in a worker process."""
    from . import noise
    from .engines import run

    snap = REPO / "data" / SNAPSHOT[t["day"]]
    noise.latest_snapshot = lambda: snap          # engines.run builds NoiseParams from the "latest" snapshot
    comp, seed = t["compiled"], t["seed"]
    t0 = time.time()
    r = run(t["engine"], comp, t["scenario"], SHOTS, seed)
    names = comp["names"]
    errors = noise.NoiseParams(t["scenario"], snap).readout_errors(
        [names[q] for q, _ in sorted(comp["measure"], key=lambda m: m[1])])
    samples = noise.apply_readout_counts(np.asarray(r["samples"], dtype=np.int64), errors,
                                         np.random.default_rng(seed + 1))
    K = len(comp["measure"])
    vals, cnt = np.unique(samples, return_counts=True)
    counts = {format(int(v), f"0{K}b"): int(c) for v, c in zip(vals, cnt)}      # bit k = char K-1-k
    return {"key": t["key"], "day": t["day"], "engine": t["engine"], "scenario": t["scenario"],
            "p": t["p"], "K": K, "n": comp["n"], "seed": seed, "snapshot": snap.name, "shots": SHOTS,
            "counts": counts, "sim_s": round(time.time() - t0, 2)}


def tasks(smoke=False):
    rows = [json.loads(line) for line in H3.read_text().splitlines()]
    if smoke:
        rows = [r for r in rows if r["day"] == "2026-11-06"][:2]
    rng = np.random.default_rng(2026_11_07)
    out = [{"key": r["key"], "day": r["day"], "p": r["p"], "compiled": r["compiled"], "engine": e,
            "scenario": s, "seed": int(rng.integers(2 ** 31))}
           for r in rows for e in ENGINES for s in SCENARIOS]
    return sorted(out, key=lambda t: -t["compiled"]["n"])          # largest first, so the pool ends evenly


def run_all(jobs, smoke):
    todo = tasks(smoke)
    out = OUT.with_name("h3_replay_smoke.jsonl") if smoke else OUT
    done = {task_key(json.loads(line)) for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [t for t in todo if task_key(t) not in done]
    print(f"h3 replay: {len(todo)} simulations to go ({len(done)} done), {jobs} processes", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(jobs) as pool, out.open("a") as f:
        futs = {pool.submit(_simulate, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:                                   # record and go on
                row = {k: t[k] for k in ("key", "day", "engine", "scenario", "p")} | {"error": repr(exc)[:400]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)


def tvd(a: dict, b: dict) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in a.keys() | b.keys())


def energies(row) -> np.ndarray:
    """QUBO energy of every outcome of a window, indexed by int(bitstring) (bit k = variable k)."""
    var = {v: k for k, v in enumerate(row["variables"])}
    idx = np.arange(2 ** len(var))
    x = (idx[:, None] >> np.arange(len(var))) & 1
    e = np.zeros(len(idx))
    for a, b, w in row["Q"]:
        e += w * (x[:, var[a]] if a == b else x[:, var[a]] * x[:, var[b]])
    return e


def summary(counts: dict, e: np.ndarray) -> dict:
    """Normalised mean energy (0 = optimum, 1 = worst), hit rate of the optimum, mean Hamming weight."""
    idx = np.array([int(k, 2) for k in counts])
    c = np.array(list(counts.values()), dtype=float)
    c /= c.sum()
    lo, hi = e.min(), e.max()
    opt = np.isclose(e[idx], lo)
    w = np.array([bin(i).count("1") for i in idx])
    return {"e": float((c * (e[idx] - lo)).sum() / (hi - lo)), "hit": float(c[opt].sum()),
            "found": bool(opt.any()), "w": float((c * w).sum())}


def analyse():
    import gzip

    from .export import OUT as CIRCUITS

    windows = {r["key"]: r for r in json.load(gzip.open(CIRCUITS, "rt"))["circuits"]}
    dev = {(r["key"], r["day"]): r for r in map(json.loads, H3.read_text().splitlines())}
    allrows = [json.loads(line) for line in OUT.read_text().splitlines()]
    sims = [r for r in allrows if "error" not in r]
    errors = [r for r in allrows if "error" in r]
    days = sorted(SNAPSHOT)
    E = {k: energies(windows[k]) for k in {k for k, _ in dev}}
    hw = {kd: summary(r["counts"], E[kd[0]]) for kd, r in dev.items()}
    floor = {k: {"tvd": tvd(dev[(k, days[0])]["counts"], dev[(k, days[1])]["counts"]),
                 "de": abs(hw[(k, days[0])]["e"] - hw[(k, days[1])]["e"]),
                 "dhit": abs(hw[(k, days[0])]["hit"] - hw[(k, days[1])]["hit"])} for k, _ in dev if _ == days[0]}
    rows = []
    for s in sims:
        k, d = s["key"], s["day"]
        sm, h = summary(s["counts"], E[k]), hw[(k, d)]
        rows.append({**{f: s[f] for f in ("key", "day", "engine", "scenario", "p", "K")},
                     "tvd": tvd(s["counts"], dev[(k, d)]["counts"]), "tvd_floor": floor[k]["tvd"],
                     "e_sim": sm["e"], "e_hw": h["e"], "de_floor": floor[k]["de"],
                     "hit_sim": sm["hit"], "hit_hw": h["hit"], "dhit_floor": floor[k]["dhit"],
                     "found_sim": sm["found"], "found_hw": h["found"], "w_sim": sm["w"], "w_hw": h["w"]})
    m = lambda g, f: np.mean([f(r) for r in g])
    fl = list(floor.values())
    lines = ["# H3: simulator vs Sirius on the circuits that ran (4096 shots each, same-day calibration)", "",
             f"{len(floor)} circuits x 2 days. Device day 1 vs day 2: mean TVD {np.mean([f['tvd'] for f in fl]):.3f}, "
             f"mean |dE| {np.mean([f['de'] for f in fl]):.3f}, mean |d hit| {np.mean([f['dhit'] for f in fl]):.4f}.",
             "E = mean normalised QUBO energy (0 optimum, 1 worst); hit = share of shots on the optimum; "
             "w = mean Hamming weight; floor = the same quantity between the two device days.", "",
             "| engine | sc. | p | TVD | TVD floor | E sim | E hw | abs dE | dE floor | hit sim | hit hw "
             "| opt found sim/hw | w sim | w hw |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in ENGINES:
        for sc in SCENARIOS:
            for p in (1, 2):
                g = [r for r in rows if r["engine"] == e and r["scenario"] == sc and r["p"] == p]
                if not g:
                    continue
                lines.append(
                    f"| {e} | {sc} | {p} | {m(g, lambda r: r['tvd']):.3f} | {m(g, lambda r: r['tvd_floor']):.3f} "
                    f"| {m(g, lambda r: r['e_sim']):.3f} | {m(g, lambda r: r['e_hw']):.3f} "
                    f"| {m(g, lambda r: abs(r['e_sim'] - r['e_hw'])):.3f} | {m(g, lambda r: r['de_floor']):.3f} "
                    f"| {m(g, lambda r: r['hit_sim']):.4f} | {m(g, lambda r: r['hit_hw']):.4f} "
                    f"| {sum(r['found_sim'] for r in g)}/{sum(r['found_hw'] for r in g)} of {len(g)} "
                    f"| {m(g, lambda r: r['w_sim']):.2f} | {m(g, lambda r: r['w_hw']):.2f} |")
    if errors:
        lines += ["", f"{len(errors)} simulations failed: " + "; ".join(sorted({r['error'][:80] for r in errors}))]
    md = OUT.with_name("h3_replay.md")
    md.write_text("\n".join(lines) + "\n")
    OUT.with_name("h3_replay_tvd.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="two circuits of day 1 on every engine/scenario")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "QULACS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if args.analyse:
        analyse()
    else:
        run_all(args.jobs, args.smoke)


if __name__ == "__main__":
    main()
