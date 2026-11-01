"""Read smaller time budgets off one long run (k-bonacci paper, PLAN.md A.2).

ILS and SA use the time limit only as a stopping rule, so a run with budget t is
a prefix of the 10 s run with the same seed. The loop tests the clock BEFORE a
move, so the incumbent for budget t is the best solution after the last move
that STARTED before t (a move that starts before t and ends after it counts,
exactly as in an independent run). Move k starts when move k-1 ends.

  extract : <src> with tl=<long>ms runs  ->  <out> with one result.json per budget
            (same directory naming and fields as the runner, so the existing
            aggregation in src/experiments/aggregate.py works unchanged)
  compare : extracted values vs independent runs (validation stage)

Usage:
    python scripts/extract_budgets.py extract --src results/kbonacci/main --out results/kbonacci/main_budgets
    python scripts/extract_budgets.py compare --extracted results/kbonacci/validate_budgets \
        --independent results/kbonacci/validate --repeat results/kbonacci/validate_repeat
    # replication control (gate verdict not applicable, read RPD and iteration columns):
    python scripts/extract_budgets.py compare --extracted results/kbonacci/replicate_budgets \
        --independent <old campaign dir>
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BUDGETS_MS = [100, 500, 1000, 2000, 5000, 10000]
TL_RE = re.compile(r"__tl=(\d+)ms__")


def load_results(root: Path) -> List[Tuple[Path, dict]]:
    """All readable result.json files under root; unreadable ones (e.g. a run killed
    while writing) are listed so the run directory can be deleted and re-run."""
    out, bad = [], []
    for p in sorted(root.rglob("result.json")):
        try:
            out.append((p, json.loads(p.read_text())))
        except (json.JSONDecodeError, OSError):
            bad.append(p)
    for p in bad:
        print(f"[results] CORRUPT (delete this directory and resume): {p.parent}")
    return out


def value_at(result: dict, t_ms: int) -> Optional[dict]:
    """Incumbent of the run as an independent run with budget t_ms would have returned it."""
    log: List[Tuple[int, int]] = [tuple(x) for x in (result.get("move_log") or [])]
    hist_c = result.get("cmax_history") or []
    hist_t = result.get("time_history_ms") or []
    if not hist_c:
        return None
    initial = hist_c[0]
    k_started = 0  # number of moves started before t
    prev_end = 0
    for end, _best in log:
        if prev_end < t_ms:
            k_started += 1
            prev_end = end
        else:
            break
    if k_started == 0:
        best, end_time = initial, 0
    else:
        end_time, best = log[k_started - 1]
    # improvements are logged once each, so the incumbent appears exactly once
    ttb = next((tt for c, tt in zip(hist_c, hist_t) if c == best and tt <= end_time), 0)
    return {"cmax_best": best, "iterations": k_started, "total_time_ms": end_time,
            "time_to_best_ms": ttb, "tl_exceeded": end_time > t_ms,
            "overrun_ms": max(0, end_time - t_ms)}


def extract(src: Path, out: Path, budgets: List[int], from_tl: Optional[int] = None) -> None:
    """Write one result.json per budget, derived ONLY from the runs with budget ``from_tl``.

    ``from_tl`` defaults to the longest budget present in ``src``. This matters for
    the validation stage, whose directory also holds the independent short runs:
    they must not be used as sources, otherwise they would overwrite the values
    read off the long runs and the validation would compare runs with themselves.
    """
    runs = load_results(src)
    all_files = [p for p, _ in runs]
    if not runs:
        print(f"[extract] no result.json under {src}")
        return
    if from_tl is None:
        from_tl = max(r["config"]["time_limit_ms"] for _, r in runs)
    runs = [(p, r) for p, r in runs if r["config"]["time_limit_ms"] == from_tl]
    files = [p for p, _ in runs]
    print(f"[extract] source budget {from_tl} ms: {len(runs)} of {len(all_files)} runs under {src}")
    n_written = mismatch = skipped = 0
    for p, r in runs:
        long_tl = r["config"]["time_limit_ms"]
        if not r.get("move_log"):
            skipped += 1
            continue
        # sanity: re-extracting the run's own budget must give its own result
        own = value_at(r, long_tl)
        if own is None or own["cmax_best"] != r["cmax_best"]:
            mismatch += 1
        lb = r.get("lower_bound")
        for t in budgets:
            if t > long_tl:
                continue
            v = value_at(r, t)
            if v is None:
                continue
            d = {k: r.get(k) for k in ("instance_jobs", "instance_machines", "upper_bound",
                                       "lower_bound", "qpu_stats")}
            d.update(v)
            d["config"] = dict(r["config"], time_limit_ms=t)
            d["gap_percent"] = (100.0 * (v["cmax_best"] - lb) / lb) if lb else None
            d["avg_iter_ms"] = (v["total_time_ms"] / v["iterations"]) if v["iterations"] else None
            d["extracted_from_tl_ms"] = long_tl
            d["source_run"] = str(p.resolve())
            d["used_qpu_fallback"] = r.get("used_qpu_fallback", False)
            name = TL_RE.sub(f"__tl={t}ms__", p.parent.name)
            target = out / name / "result.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(d, indent=1))
            n_written += 1
    print(f"[extract] {len(files)} source runs, {n_written} budget results written to {out}")
    if skipped:
        print(f"[extract] WARNING: {skipped} runs without move_log skipped (old code?)")
    if mismatch:
        print(f"[extract] WARNING: {mismatch} runs where re-extraction of the own budget "
              "differs from cmax_best -- inspect before using the data")
    else:
        print("[extract] self-check OK: own-budget re-extraction equals cmax_best for every run")


# same legacy-name normalization as src/experiments/aggregate.py ("fibonahi" typo in old batches)
_NEIGH_RENAMES = {"fibonahi": "fibonacci", "quantum_fibonahi": "quantum_fibonacci"}


def _gap(r: dict) -> float:
    """RPD to the lower bound; recomputed when an (old) result lacks gap_percent."""
    g = r.get("gap_percent")
    if g is None and r.get("lower_bound"):
        g = 100.0 * (r["cmax_best"] - r["lower_bound"]) / r["lower_bound"]
    return g


def _key(r: dict) -> tuple:
    c = r["config"]
    neigh = _NEIGH_RENAMES.get(c["neighborhood"], c["neighborhood"])
    return (c["algorithm"], neigh, Path(c["instance_file"]).stem,
            c["instance_number"], c["seed"], c["time_limit_ms"])


def _abs_diff_noise(independent: Path, repeat: Path) -> Dict[tuple, float]:
    """Run-to-run noise: mean |RPD difference| between two independent runs of the same config."""
    a = {_key(r): _gap(r) for _, r in load_results(independent)}
    b = {_key(r): _gap(r) for _, r in load_results(repeat)}
    g: Dict[tuple, List[float]] = {}
    for k, v in b.items():
        if k in a:
            g.setdefault((k[0], k[1], k[5]), []).append(abs(a[k] - v))
    return {k: statistics.mean(v) for k, v in g.items()}


GATE_ABS_PP = 0.10  # always tolerated (one move of timing jitter at tiny budgets)


def compare(extracted: Path, independent: Path, repeat: Optional[Path] = None) -> bool:
    """Print the validation table and return the gate verdict.

    Gate per row (algorithm, neighborhood, budget): mean |extracted - independent|
    <= max(0.10 pp, 2 x run-to-run noise), where the noise is the mean |difference|
    between two independent runs of the same configuration (``repeat``).
    """
    ext = {_key(r): r for _, r in load_results(extracted)}
    ind = {_key(r): (p, r) for p, r in load_results(independent)}
    groups: Dict[tuple, List[Tuple[float, float]]] = {}
    iters: Dict[tuple, List[Tuple[int, int]]] = {}
    for k, (p_ind, r) in ind.items():
        # skip only the source run itself (validation: the 10 s run in the same directory);
        # an independent run with the same budget (replication: old campaign) is compared
        if k in ext and str(p_ind.resolve()) != ext[k].get("source_run"):
            groups.setdefault((k[0], k[1], k[5]), []).append((_gap(ext[k]), _gap(r)))
            iters.setdefault((k[0], k[1], k[5]), []).append((ext[k].get("iterations") or 0,
                                                             r.get("iterations") or 0))
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    noise = _abs_diff_noise(independent, repeat) if repeat else {}
    print(f"{'algo':<4} {'neigh':<14} {'tl':>6} {'n':>4} {'RPD extr.':>10} {'RPD indep.':>11} "
          f"{'diff':>7} {'|diff|':>7} {'noise |diff|':>12} {'p (Wilcoxon)':>13} {'gate':>5} "
          f"{'iters extr.':>11} {'iters indep.':>12}")
    failed = []
    for (algo, neigh, tl), pairs in sorted(groups.items()):
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        diff = statistics.mean(x - y for x, y in pairs)
        p = float("nan")
        if wilcoxon and any(x != y for x, y in pairs):
            p = wilcoxon(a, b).pvalue
        absd = statistics.mean(abs(x - y) for x, y in pairs)
        nz = noise.get((algo, neigh, tl))
        nz_s = f"{nz:12.2f}" if nz is not None else f"{'-':>12}"
        ok = absd <= max(GATE_ABS_PP, 2 * nz if nz is not None else 0.0)
        if not ok:
            failed.append((algo, neigh, tl))
        print(f"{algo:<4} {neigh:<14} {tl:>6} {len(pairs):>4} {statistics.mean(a):>10.2f} "
              f"{statistics.mean(b):>11.2f} {diff:>7.2f} {absd:>7.2f} {nz_s} {p:>13.3f} "
              f"{'ok' if ok else 'FAIL':>5} "
              f"{statistics.mean(x for x, _ in iters[(algo, neigh, tl)]):>11.0f} "
              f"{statistics.mean(y for _, y in iters[(algo, neigh, tl)]):>12.0f}")
    print("Expected: ILS |diff| ~ 0 (deterministic given the seed); SA |diff| of the order of the")
    print("run-to-run noise column (SA reheats on a wall-clock stagnation timer, so even two runs")
    print("with the same seed and budget diverge after the first kick).")
    if not groups:
        print("VALIDATION GATE: FAIL (no comparable pairs found)")
        return False
    if failed:
        print(f"VALIDATION GATE: FAIL in {len(failed)} rows: {failed}")
        return False
    print(f"VALIDATION GATE: PASS (|diff| <= max({GATE_ABS_PP} pp, 2 x run-to-run noise) in every row)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--src", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--budgets", default=",".join(map(str, BUDGETS_MS)))
    e.add_argument("--from-tl", type=int, default=None,
                   help="source budget in ms (default: the longest budget found under --src)")
    c = sub.add_parser("compare")
    c.add_argument("--extracted", type=Path, required=True)
    c.add_argument("--independent", type=Path, required=True)
    c.add_argument("--repeat", type=Path, default=None,
                   help="second independent copy of some runs (run-to-run noise control)")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.src, a.out, [int(x) for x in a.budgets.split(",")], a.from_tl)
    else:
        sys.exit(0 if compare(a.extracted, a.independent, a.repeat) else 1)


if __name__ == "__main__":
    main()
