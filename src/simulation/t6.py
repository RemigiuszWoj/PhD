"""T6: time and memory of every simulator versus window size K and depth p.

``export`` (.venv-iqm) takes the first-move windows (identity start) of one
instance of every size class, tai20_5 ... tai500_20, groups them by K and keeps
up to ``--per-k`` windows for every K from 4 to 16, taken round-robin over
the four neighborhoods. Each window is compiled for p = 1..6 with the "small"
angle table (the only one that reaches p = 6; angles do not change the cost).

``run`` (.venv-annsim) measures one (engine, scenario, p, K) batch at a time,
strictly one after another, so the times are not disturbed by other cells.
A window that takes longer than ``--cap`` seconds stops the series: larger K at
the same (engine, scenario, p) are recorded as "above cap" without running, so
the largest K under the cap is the "maximum feasible K" of that program. The
run stops launching batches after ``--budget-h`` hours; it resumes where it
stopped. No references are made (agreement is stage 1 and 2's job).
``--threads N`` repeats the measurement with N threads per process.

Usage:
    PYTHONHASHSEED=0 PYTHONPATH=. .venv-iqm/bin/python -m src.simulation.t6 export
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.t6 run [--cap 300] [--budget-h 10] [--threads 1]
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.t6 summary
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import tempfile
import time
from pathlib import Path

from .noise import REPO

MEAS = REPO / "articles" / "annsim_simulators" / "measurements"
CIRCUITS = MEAS / "circuits_t6.json.gz"
OUT = MEAS / "t6_scaling.jsonl"
CLASSES = ["tai20_5", "tai50_5", "tai100_5", "tai200_10", "tai500_20"]
K_RANGE = range(4, 17)
P_RANGE = (1, 2, 3, 4, 5, 6)
NEIGHBORHOODS = ("adjacent", "fibonacci", "dynasearch", "motzkin")


def export(per_k: int, out: Path):
    from src.experiments.qaoa_depth_study import capture_windows, energies
    from src.neighborhoods.gate_qaoa.fake_sirius import fake_sirius, latest_snapshot
    from src.parser import parser

    from .export import compile_window

    by_k = collections.defaultdict(lambda: collections.defaultdict(list))   # K -> neigh -> windows
    for f in CLASSES:
        d = parser(str(REPO / "data" / f"{f}.txt"), 0)
        n, pt = d["info"]["jobs"], d["processing_times"]
        for nb in NEIGHBORHOODS:
            for wi, Q in enumerate(capture_windows(nb, list(range(n)), pt)):
                K = len({v for pair in Q for v in pair})
                if K in K_RANGE:
                    by_k[K][nb].append((f, nb, wi, Q))
    fake = fake_sirius(mode="calibrated")
    rows = []
    for K in K_RANGE:
        pick, queues = [], [list(by_k[K][nb]) for nb in NEIGHBORHOODS]
        while len(pick) < per_k and any(queues):                  # round-robin over neighborhoods
            for q in queues:
                if q and len(pick) < per_k:
                    pick.append(q.pop(0))
        for f, nb, wi, Q in pick:
            variables = sorted({v for pair in Q for v in pair})
            E = energies(Q, variables)
            for p in P_RANGE:
                var2, circ = compile_window(Q, "small", nb, p, fake)
                assert var2 == variables
                rows.append({"key": f"t6|{f}|0|{nb}|{wi}|small|{p}", "file": f, "inst": 0, "neigh": nb,
                             "window": wi, "table": "small", "p": p, "K": K, "variables": variables,
                             "Q": [[a, b, w] for (a, b), w in Q.items()], "E_min": float(E.min()), **circ})
            print(f"K={K:2d} {f} {nb} window {wi}: p=1 cz={rows[-6]['compiled']['cz']} "
                  f"move={rows[-6]['compiled']['move']}", flush=True)
        if len(pick) < per_k:
            print(f"K={K}: only {len(pick)} windows in {CLASSES}", flush=True)
    meta = {"snapshot": latest_snapshot().name, "classes": CLASSES, "per_k": per_k, "table": "small",
            "compiler": "qiskit-iqm transpile_to_IQM, optimization_level=1, seed_transpiler=0",
            "made": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with gzip.open(out, "wt") as fh:
        json.dump({"_meta": meta, "circuits": rows}, fh)
    print(f"{len(rows)} circuits -> {out}")


def run(engines, scenarios, ps, cap: float, budget_h: float, threads: int, shots: int, out: Path):
    from .engines import ENGINES, unsupported
    from .stage1 import DM_ALWAYS, DM_ENGINES, DM_MAX_QUBITS, circuits
    from .stage2 import _batch_call

    cs = circuits(CIRCUITS)
    ref_dir = MEAS / "refs_t6"                       # stays empty: no agreement check in T6
    ref_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="annsim_t6_"))
    done, stopped = set(), {}
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            done.add((r["engine"], r["scenario"], r["p"], r["K"], r.get("threads", 1)))
            if r.get("above_cap"):
                stopped.setdefault((r["engine"], r["scenario"], r["p"], r.get("threads", 1)), r["K"])
    os.environ["ANNSIM_THREADS"] = str(threads)
    t_end = time.time() + budget_h * 3600
    by = collections.defaultdict(list)
    for k, c in cs.items():
        by[(c["p"], c["K"])].append(k)
    with out.open("a") as fh:
        for p in ps:
            for s in scenarios:
                for e in (x for x in ENGINES if x in engines):
                    series = (e, s, p, threads)
                    for K in K_RANGE:
                        keys = by.get((p, K), [])
                        if not keys or (e, s, p, K, threads) in done:
                            continue
                        base = {"engine": e, "scenario": s, "p": p, "K": K, "threads": threads,
                                "n_qubits": cs[keys[0]]["compiled"]["n"], "windows": len(keys)}
                        why = unsupported(e, s)
                        if not why and (e in DM_ALWAYS or (s != "S1" and e in DM_ENGINES)) \
                                and base["n_qubits"] > DM_MAX_QUBITS:
                            why = f"density matrix > {DM_MAX_QUBITS} qubits"
                        if why or series in stopped:
                            row = {**base, "ok": False, "skipped": why or f"above cap from K={stopped[series]}"}
                            if not why:
                                row["above_cap"] = True
                            fh.write(json.dumps(row) + "\n")
                            continue
                        if time.time() > t_end:
                            print(f"budget of {budget_h} h used up; rerun to continue", flush=True)
                            return
                        kf = tmp / f"{e}_{s}_{p}_{K}.txt"
                        kf.write_text("\n".join(keys))
                        t0 = time.time()
                        rows, err, wall = _batch_call(
                            ["--keys-file", str(kf), "--engine", e, "--which", "compiled", "--scenario", s,
                             "--shots", str(shots), "--circuits", str(CIRCUITS), "--ref-dir", str(ref_dir)],
                            timeout=int(cap * len(keys) + 120))
                        times = [r["t_run"] + r["t_build"] for r in rows if r.get("ok")]
                        slow = err or len(times) < len(keys) or max(times) > cap
                        row = {**base, "ok": not slow, "error": err, "wall_s": time.time() - t0,
                               "t": times, "t_median": sorted(times)[len(times) // 2] if times else None,
                               "peak_rss_mb": max((r.get("peak_rss_mb", 0) for r in rows), default=None),
                               "methods": sorted({r.get("method", "") for r in rows if r.get("ok")})}
                        if slow:
                            row["above_cap"] = True
                            stopped[series] = K
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(f"p={p} {s} {e:11s} K={K:2d}: "
                              f"{'ABOVE CAP' if slow else 'ok'} median {row['t_median']} s, "
                              f"RSS {row['peak_rss_mb']} MB {err or ''}"[:200], flush=True)


def summary(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    g = collections.defaultdict(dict)
    for r in rows:
        g[(r["engine"], r["scenario"], r["p"], r.get("threads", 1))][r["K"]] = r
    print("| engine | sc | p | thr | max K under cap | t [s] at K=8 / 12 / 14 / 16 | RSS max [MB] | note |")
    print("|---|---|---|---|---|---|---|---|")
    for (e, s, p, th), ks in sorted(g.items(), key=lambda x: (x[0][2], x[0][1], x[0][0], x[0][3])):
        ok = [K for K, r in ks.items() if r.get("ok")]
        t = lambda K: (f"{ks[K]['t_median']:.3g}" if K in ks and ks[K].get("ok") else "–")
        note = next((r.get("skipped") for r in ks.values() if r.get("skipped") and not r.get("above_cap")), "")
        print(f"| {e} | {s} | {p} | {th} | {max(ok) if ok else '–'} | {t(8)} / {t(12)} / {t(14)} / {t(16)} | "
              f"{max((r.get('peak_rss_mb') or 0 for r in ks.values()), default=0):.0f} | {note[:60]} |")


def main():
    from .engines import ENGINES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--per-k", type=int, default=3)
    ex.add_argument("--out", type=Path, default=CIRCUITS)
    r = sub.add_parser("run")
    r.add_argument("--engines", nargs="*", default=list(ENGINES))
    r.add_argument("--scenarios", nargs="*", default=["S1", "S2", "S3"])
    r.add_argument("--p", nargs="*", type=int, default=list(P_RANGE))
    r.add_argument("--cap", type=float, default=300.0, help="seconds per window above which a series stops")
    r.add_argument("--budget-h", type=float, default=10.0)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--shots", type=int, default=4096)
    r.add_argument("--out", type=Path, default=OUT)
    sm = sub.add_parser("summary")
    sm.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.per_k, args.out)
    elif args.cmd == "run":
        run(set(args.engines), args.scenarios, args.p, args.cap, args.budget_h, args.threads, args.shots,
            args.out)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
