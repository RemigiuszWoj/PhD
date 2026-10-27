"""Stage 2: the basic Taillard set (tai20_5/10/20, instances 0-3) on every engine that can run it.

The circuits are the 161 first-move windows with an improving move (export.py),
here those with K <= ``--kmax`` (71 windows for K <= 11, so that density
matrices stay at <= 12 qubits) at p = 1..3. As in stage 1, every engine runs in
its own process, one thread, but a process takes a whole batch of circuits
(``--batch``) so the import cost of the heavier packages (cirq, cudaq, qibo)
is paid once per batch; ``peak_rss_mb`` is then the running maximum of the
batch. Cells that stage 1 showed to be unsupported are not run, but listed
with the reason, so the table of the paper can say why a cell is empty.

Usage (from .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.stage2 --scenarios S1 [--jobs 8]
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.stage2 --estimate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .engines import ENGINES, unsupported
from .export import OUT as CIRCUITS
from .stage1 import DM_ALWAYS, DM_ENGINES, DM_MAX_QUBITS, MEAS, OUT as OUT1, cell_key, circuits, ref_dir_for

OUT = MEAS / "stage2_n20.jsonl"
def select(cs, kmax, ps):
    keys = [k for k, c in cs.items() if c["K"] <= kmax and c["p"] in ps]
    return sorted(keys, key=lambda k: (cs[k]["p"], cs[k]["K"], k))


def _batch_call(args, timeout):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "src.simulation.cell", *args],
                           cwd=MEAS.parents[2], capture_output=True, text=True, timeout=timeout)
        out, err = p.stdout, f"exit {p.returncode}: {(p.stderr or '')[-600:]}" if p.returncode else None
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = f"timeout after {timeout} s"
    rows = [json.loads(line[6:]) for line in out.splitlines() if line.startswith("@@ROW ")]
    return rows, err, time.time() - t0


def _chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def run(keys, engines, scenarios, shots, jobs, batch, timeout, out: Path, cpath: Path):
    cs = circuits(cpath)
    ref_dir = ref_dir_for(cpath)
    ref_dir.mkdir(parents=True, exist_ok=True)
    common = ["--circuits", str(cpath), "--ref-dir", str(ref_dir)]
    tmp = Path(tempfile.mkdtemp(prefix="annsim_stage2_"))

    def keyfile(ks):
        f = tmp / f"keys_{abs(hash(tuple(ks)))}.txt"
        f.write_text("\n".join(ks))
        return str(f)

    # references: S1 logical statevector, S2/S3 Aer density matrix (missing ones only)
    todo_ref = [(k, s) for s in scenarios for k in keys
                if not (ref_dir / (k.replace("|", "_") + f"_{s}.npy")).exists()]
    by_s = {s: [k for k, s2 in todo_ref if s2 == s] for s in scenarios}
    ref_jobs = [(s, ch) for s, ks in by_s.items() for ch in _chunks(ks, batch)]
    print(f"{len(todo_ref)} references to make in {len(ref_jobs)} batches", flush=True)

    def one_ref(job):
        s, ch = job
        rows, err, wall = _batch_call(["--keys-file", keyfile(ch), "--scenario", s, *common, "--make-ref"], timeout)
        return s, len(ch), len(rows), err, wall

    with ThreadPoolExecutor(jobs) as pool:
        for s, n, got, err, wall in pool.map(one_ref, ref_jobs):
            print(f"ref {s}: {got}/{n} in {wall:.0f} s {err or ''}"[:300], flush=True)

    done = set()
    for path in (out,):
        if path.exists():
            done |= {cell_key(r["engine"], r["key"], r["which"], r["scenario"])
                     for r in map(json.loads, path.read_text().splitlines())}
    jobs_list, notes = [], []
    for s in scenarios:
        for e in engines:
            why = unsupported(e, s)
            ks = [k for k in keys if cell_key(e, k, "compiled", s) not in done]
            if why:
                notes += [{"engine": e, "key": k, "which": "compiled", "scenario": s, "K": cs[k]["K"],
                           "p": cs[k]["p"], "ok": False, "skipped": why} for k in ks]
                continue
            if e in DM_ALWAYS or (s != "S1" and e in DM_ENGINES):
                big = [k for k in ks if cs[k]["compiled"]["n"] > DM_MAX_QUBITS]
                notes += [{"engine": e, "key": k, "which": "compiled", "scenario": s, "K": cs[k]["K"],
                           "p": cs[k]["p"], "ok": False, "skipped": f"density matrix > {DM_MAX_QUBITS} qubits"}
                          for k in big]
                ks = [k for k in ks if k not in big]
            jobs_list += [(e, s, ch) for ch in _chunks(ks, batch)]
    print(f"{sum(len(j[2]) for j in jobs_list)} cells in {len(jobs_list)} batches, {jobs} at a time", flush=True)

    def one(job):
        e, s, ch = job
        rows, err, wall = _batch_call(["--keys-file", keyfile(ch), "--engine", e, "--which", "compiled",
                                       "--scenario", s, "--shots", str(shots), *common], timeout)
        got = {r["key"] for r in rows}
        for k in ch:
            if k not in got:                     # the batch died on this key (or before it)
                rows.append({"engine": e, "key": k, "which": "compiled", "scenario": s, "K": cs[k]["K"],
                             "p": cs[k]["p"], "ok": False, "error": err or "no row"})
        return e, s, rows, err, wall

    with ThreadPoolExecutor(jobs) as pool, out.open("a") as fh:
        for r in notes:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        for e, s, rows, err, wall in pool.map(one, jobs_list):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            ok = sum(1 for r in rows if r.get("ok"))
            print(f"{e:10s} {s}: {ok}/{len(rows)} ok in {wall:.0f} s {err or ''}"[:300], flush=True)


def estimate(cs, keys, stage1_path: Path):
    """Extrapolate the cost of every (engine, scenario) from the stage-1 times (gates x 2^n scaling)."""
    rows = [json.loads(line) for line in stage1_path.read_text().splitlines() if line]
    import collections

    rate = collections.defaultdict(list)            # seconds per (gate x 2^n) or (gate x 4^n) for DM
    for r in rows:
        if not r.get("ok") or r["which"] != "compiled":
            continue
        c = cs[r["key"]]
        n, g = c["compiled"]["n"], len(c["compiled"]["ops"])
        dm = "density matrix" in (r.get("method") or "")
        traj = "trajector" in (r.get("method") or "")
        work = g * (4 ** n if dm else 2 ** n) * (r["shots"] if traj else 1)
        rate[(r["engine"], r["scenario"])].append((r["t_run"] + r["t_build"]) / work)
    total = collections.defaultdict(float)
    for (e, s), v in rate.items():
        r_med = sorted(v)[len(v) // 2]
        for k in keys:
            c = cs[k]
            n, g = c["compiled"]["n"], len(c["compiled"]["ops"])
            meth = "dm" if (e in DM_ALWAYS or (s != "S1" and e in DM_ENGINES)) else "sv"
            traj = s != "S1" and e in ("aer_sv", "qsim", "qulacs", "ddsim", "qibo")
            total[(e, s)] += r_med * g * (4 ** n if meth == "dm" else 2 ** n) * (4096 if traj else 1)
    for (e, s), t in sorted(total.items(), key=lambda x: -x[1]):
        print(f"{e:10s} {s}: ~{t / 3600:7.2f} h single-thread for {len(keys)} circuits")


def summary(path: Path):
    """Per engine and scenario: cells run / ok, agreement with the reference, time by K, peak memory.

    Exact methods: max TVD to the reference. Sampled methods: share of cells whose TVD lies within the
    95th percentile of the shot-noise floor (over assignments where that floor is <= 0.3, otherwise over
    QUBO energy levels); with 5 % expected outside by chance.
    """
    import collections
    import statistics

    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    last = {}
    for r in rows:                                   # a rerun of a cell replaces the earlier row
        last[cell_key(r["engine"], r["key"], r["which"], r["scenario"])] = r
    g = collections.defaultdict(list)
    for r in last.values():
        g[(r["engine"], r["scenario"])].append(r)
    print(f"{'engine':10s} sc  run  ok  skip  exact:max TVD  sampled:in floor   t median [s] K<=6 / 9-11   "
          f"RSS max [MB]  note")
    for (e, sc), rs in sorted(g.items(), key=lambda x: (x[0][1], ENGINES.index(x[0][0]))):
        ok = [r for r in rs if r.get("ok")]
        skip = [r for r in rs if r.get("skipped")]
        ex = [r["tvd_exact"] for r in ok if r.get("tvd_exact") is not None]
        inside, n_s = 0, 0
        for r in ok:
            if r.get("tvd_exact") is None and r.get("tvd_shots") is not None:
                n_s += 1
                full = r["floor_p95"] <= 0.3
                inside += (r["tvd_shots"] <= r["floor_p95"]) if full else (r["tvd_E_shots"] <= r["floor_E_p95"])
        t_small = [r["t_run"] + r["t_build"] for r in ok if r["K"] <= 6]
        t_big = [r["t_run"] + r["t_build"] for r in ok if r["K"] >= 9]
        med = lambda v: f"{statistics.median(v):8.3g}" if v else "       -"
        err = collections.Counter((r.get("error") or "")[:60] for r in rs if not r.get("ok") and not r.get("skipped"))
        note = (skip[0]["skipped"][:50] if skip else "") + (f" ERR {dict(err)}" if err else "")
        print(f"{e:10s} {sc} {len(rs) - len(skip):4d} {len(ok):3d} {len(skip):5d}  "
              f"{(f'{max(ex):.1e}' if ex else '-'):>13s}  {(f'{inside}/{n_s}' if n_s else '-'):>16s}   "
              f"{med(t_small)} / {med(t_big)}   {max((r.get('peak_rss_mb', 0) for r in ok), default=0):10.0f}  {note}"[:220])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kmax", type=int, default=11)
    ap.add_argument("--p", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--engines", nargs="*", default=list(ENGINES))
    ap.add_argument("--scenarios", nargs="*", default=["S1", "S2", "S3"])
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--circuits", type=Path, default=CIRCUITS)
    ap.add_argument("--estimate", action="store_true", help="cost extrapolated from stage 1, run nothing")
    ap.add_argument("--summary", action="store_true", help="per engine and scenario summary of --out")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    cs = circuits(args.circuits)
    keys = select(cs, args.kmax, set(args.p))
    print(f"{len(keys)} circuits (K <= {args.kmax}, p in {sorted(set(args.p))})", flush=True)
    if args.estimate:
        estimate(cs, keys, OUT1)
        return
    run(keys, args.engines, args.scenarios, args.shots, args.jobs, args.batch, args.timeout, args.out,
        args.circuits)


if __name__ == "__main__":
    main()
