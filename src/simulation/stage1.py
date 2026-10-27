"""Stage 1 (and the S1 / small-window part of stage 2): every engine on chosen circuits.

Each cell (engine x circuit x logical|compiled x scenario) runs in its own
process (cell.py) with one thread, so a crash, a hang or a memory blow-up of
one simulator does not take the campaign down, and the peak RSS belongs to
that cell alone. Rows go to ``--out`` (JSONL); reruns skip cells already there.

Density-matrix methods are skipped above ``DM_MAX_QUBITS`` in S2/S3 (a 14-qubit
density matrix is 4 GB complex128 before any engine overhead; 24 GB RAM here).

Usage (from .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.stage1 --preset stage1 [--jobs 4]
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.stage1 --summary
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .engines import ENGINES, unsupported
from .export import OUT as CIRCUITS
from .noise import REPO

MEAS = REPO / "articles" / "annsim_simulators" / "measurements"
OUT = MEAS / "stage1_engines.jsonl"
REF_DIR = MEAS / "refs"
DM_MAX_QUBITS = 13
# Methods that hold a density matrix in S2/S3 (and therefore stop at DM_MAX_QUBITS) ...
DM_ENGINES = {"aer_dm", "cirq_dm", "qulacs_dm", "qibo_dm", "cudaq", "pennylane", "braket", "ddsim_dm"}
# ... and those that hold one even without noise.
DM_ALWAYS = {"aer_dm", "cirq_dm", "qulacs_dm", "qibo_dm"}
# One window per size class from tai20_5 instance 0 (identity start), p = 1.
PRESETS = {
    "stage1": ["tai20_5|0|fibonacci|1|small|1",      # K = 4
               "tai20_5|0|dynasearch|3|small|1",     # K = 9
               "tai20_5|0|motzkin|1|small|1",        # K = 12
               "tai20_5|0|dynasearch|0|small|1"],    # K = 15
}


def circuits(path: Path = CIRCUITS):
    with gzip.open(path, "rt") as fh:
        return {c["key"]: c for c in json.load(fh)["circuits"]}


def ref_dir_for(path: Path) -> Path:
    """S2/S3 references belong to one compiled circuit, so each export gets its own folder."""
    return REF_DIR if path == CIRCUITS else MEAS / f"refs_{path.name.split('.')[0]}"


def cell_key(engine, key, which, scenario):
    return f"{engine}|{key}|{which}|{scenario}"


def _done(path):
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out.add(cell_key(r["engine"], r["key"], r["which"], r["scenario"]) if "engine" in r else r["key"])
    return out


def _call(args, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "src.simulation.cell", *args], cwd=REPO,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout} s", time.time() - t0
    for line in p.stdout.splitlines():
        if line.startswith("@@ROW "):
            return json.loads(line[6:]), None, time.time() - t0
    return None, f"exit {p.returncode}: {(p.stderr or p.stdout)[-800:]}", time.time() - t0


def plan(keys, engines, scenarios, whiches, cs):
    cells = []
    for key in keys:
        c = cs[key]
        for scenario in scenarios:
            for which in whiches:
                if which == "logical" and scenario != "S1":
                    continue
                for e in engines:
                    skip = unsupported(e, scenario)
                    n = c[which]["n"]
                    if (e in DM_ALWAYS or (scenario != "S1" and e in DM_ENGINES)) and n > DM_MAX_QUBITS:
                        skip = f"density matrix of {n} qubits (> {DM_MAX_QUBITS})"
                    cells.append((e, key, which, scenario, skip))
    return cells


def run(keys, engines, scenarios, whiches, shots, jobs, timeout, out: Path, cpath: Path = CIRCUITS):
    cs = circuits(cpath)
    ref_dir = ref_dir_for(cpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    common = ["--circuits", str(cpath), "--ref-dir", str(ref_dir)]
    # references first: S1 logical statevector, S2/S3 Aer density matrix
    for key in keys:
        for scenario in scenarios:
            row, err, _ = _call(["--key", key, "--scenario", scenario, *common, "--make-ref"], timeout)
            print(f"ref {key} {scenario}: {row if row else err}", flush=True)
    done = _done(out)
    cells = [c for c in plan(keys, engines, scenarios, whiches, cs)
             if cell_key(*c[:4]) not in done]
    print(f"{len(cells)} cells to run, {jobs} at a time", flush=True)

    def one(cell):
        e, key, which, scenario, skip = cell
        base = {"engine": e, "key": key, "which": which, "scenario": scenario, "K": cs[key]["K"]}
        if skip:
            return {**base, "ok": False, "skipped": skip}
        row, err, wall = _call(["--key", key, "--engine", e, "--which", which, "--scenario", scenario,
                                "--shots", str(shots), *common], timeout)
        return row if row else {**base, "ok": False, "error": err, "wall_s": wall}

    with ThreadPoolExecutor(jobs) as pool, out.open("a") as fh:
        for row in pool.map(one, cells):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            status = "ok" if row.get("ok") else ("skip" if row.get("skipped") else "FAIL")
            tvd = row.get("tvd_exact", row.get("tvd_shots"))
            print(f"{row['engine']:10s} {row['which']:8s} {row['scenario']} K={row['K']:2d} {status:4s} "
                  f"t={row.get('t_run', 0):8.2f}s tvd={tvd if tvd is None else round(tvd, 4)} "
                  f"{row.get('error', '') or row.get('skipped', '') or row.get('method', '')}"[:200], flush=True)


def summary(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows.sort(key=lambda r: (r["scenario"], r["which"], ENGINES.index(r["engine"]), r["K"]))
    print(f"{'engine':10s} {'which':8s} sc K  ok   t_run[s]  RSS[MB]  tvd_exact  tvd_shots  floor_med  "
          f"P_opt(sh/ref)  method / error")
    for r in rows:
        f = lambda k, w=9, d=4: (f"{r[k]:{w}.{d}f}" if isinstance(r.get(k), (int, float)) else " " * (w - 1) + "-")
        note = r.get("method") if r.get("ok") else (r.get("skipped") or r.get("error", ""))
        print(f"{r['engine']:10s} {r['which']:8s} {r['scenario']} {r['K']:2d} {'y' if r.get('ok') else 'n':3s} "
              f"{f('t_run', 10, 3)} {f('peak_rss_mb', 8, 0)} {f('tvd_exact', 10, 6)} {f('tvd_shots', 10, 4)} "
              f"{f('floor_median', 10, 4)}  {f('P_opt_shots', 6, 3)}/{f('P_opt_ref', 6, 3)}  {str(note)[:90]}")


def table_md(path: Path):
    """Markdown table for the report: per engine and scenario, agreement and time for every K.

    Agreement: ``=`` exact distribution equal to the reference (TVD < 1e-9; ``≈`` below 1e-5: single
    precision in qsim, Aer's thermal_relaxation_error vs the Kraus form in S3), ``~`` sampled and
    within the 95th percentile of the shot-noise floor (over 2^K outcomes, or over energy levels
    where that floor is above 0.3), ``x`` outside it, ``-`` not run (reason in the notes).
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    Ks = sorted({r["K"] for r in rows})
    out = ["| engine | method | " + " | ".join(f"{s} K={k}" for s in ("S1", "S2", "S3") for k in Ks) + " |",
           "|---|---|" + "---|" * (3 * len(Ks))]
    by = {(r["engine"], r["scenario"], r["K"]): r for r in rows if r["which"] == "compiled"}
    for e in ENGINES:
        meth = sorted({(r.get("method") or "").split(" (")[0] for r in rows if r["engine"] == e and r.get("ok")})
        cells = []
        for s in ("S1", "S2", "S3"):
            for k in Ks:
                r = by.get((e, s, k))
                if r is None or not r.get("ok"):
                    cells.append("-")
                    continue
                if r.get("tvd_exact") is not None:
                    d = r["tvd_exact"]
                    tag = "=" if d < 1e-9 else (f"≈ {d:.0e}" if d < 1e-5 else f"x {d:.1e}")
                elif r.get("tvd_shots") is not None:
                    full = r.get("floor_p95", 1) <= 0.3
                    tvd, fl = (r["tvd_shots"], r["floor_p95"]) if full else (r["tvd_E_shots"], r["floor_E_p95"])
                    tag = ("~" if tvd <= fl else "x") + f" {tvd:.3f}" + ("" if full else "E")
                else:
                    tag = "?"
                t = r["t_run"] + r["t_build"]
                cells.append(f"{tag}<br>{t:.2g} s, {r.get('peak_rss_mb', 0):.0f} MB")
        out.append(f"| {e} | {', '.join(m for m in meth if m)} | " + " | ".join(cells) + " |")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--engines", nargs="*", default=list(ENGINES))
    ap.add_argument("--scenarios", nargs="*", default=["S1", "S2", "S3"])
    ap.add_argument("--which", nargs="*", default=["logical", "compiled"])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--table-md", action="store_true", help="markdown table of --out for the report")
    ap.add_argument("--circuits", type=Path, default=CIRCUITS, help="export.py (qiskit-iqm) or compile_cirq.py file")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    if args.table_md:
        table_md(args.out)
        return
    keys = args.keys or PRESETS[args.preset]
    run(keys, args.engines, args.scenarios, args.which, args.shots, args.jobs, args.timeout, args.out,
        args.circuits)


if __name__ == "__main__":
    main()
