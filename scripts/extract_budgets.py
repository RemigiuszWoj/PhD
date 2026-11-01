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
        --independent results/kbonacci/validate
"""

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BUDGETS_MS = [100, 500, 1000, 2000, 5000, 10000]
TL_RE = re.compile(r"__tl=(\d+)ms__")


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


def extract(src: Path, out: Path, budgets: List[int]) -> None:
    files = sorted(src.rglob("result.json"))
    n_written = mismatch = skipped = 0
    for p in files:
        r = json.loads(p.read_text())
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


def _key(r: dict) -> tuple:
    c = r["config"]
    return (c["algorithm"], c["neighborhood"], Path(c["instance_file"]).stem,
            c["instance_number"], c["seed"], c["time_limit_ms"])


def _abs_diff_noise(independent: Path, repeat: Path) -> Dict[tuple, float]:
    """Run-to-run noise: mean |RPD difference| between two independent runs of the same config."""
    a = {_key(json.loads(p.read_text())): json.loads(p.read_text())["gap_percent"]
         for p in independent.rglob("result.json")}
    b = {_key(json.loads(p.read_text())): json.loads(p.read_text())["gap_percent"]
         for p in repeat.rglob("result.json")}
    g: Dict[tuple, List[float]] = {}
    for k, v in b.items():
        if k in a:
            g.setdefault((k[0], k[1], k[5]), []).append(abs(a[k] - v))
    return {k: statistics.mean(v) for k, v in g.items()}


def compare(extracted: Path, independent: Path, repeat: Optional[Path] = None) -> None:
    ext = {_key(json.loads(p.read_text())): json.loads(p.read_text())
           for p in extracted.rglob("result.json")}
    ind = {_key(json.loads(p.read_text())): json.loads(p.read_text())
           for p in independent.rglob("result.json")}
    groups: Dict[tuple, List[Tuple[float, float]]] = {}
    for k, r in ind.items():
        if k in ext and r["config"]["time_limit_ms"] != ext[k].get("extracted_from_tl_ms"):
            groups.setdefault((k[0], k[1], k[5]), []).append((ext[k]["gap_percent"], r["gap_percent"]))
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    noise = _abs_diff_noise(independent, repeat) if repeat else {}
    print(f"{'algo':<4} {'neigh':<14} {'tl':>6} {'n':>4} {'RPD extr.':>10} {'RPD indep.':>11} "
          f"{'diff':>7} {'|diff|':>7} {'noise |diff|':>12} {'p (Wilcoxon)':>13}")
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
        print(f"{algo:<4} {neigh:<14} {tl:>6} {len(pairs):>4} {statistics.mean(a):>10.2f} "
              f"{statistics.mean(b):>11.2f} {diff:>7.2f} {absd:>7.2f} {nz_s} {p:>13.3f}")
    print("Expected: ILS |diff| ~ 0 (deterministic given the seed); SA |diff| of the order of the")
    print("run-to-run noise column (SA reheats on a wall-clock stagnation timer, so even two runs")
    print("with the same seed and budget diverge after the first kick).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--src", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--budgets", default=",".join(map(str, BUDGETS_MS)))
    c = sub.add_parser("compare")
    c.add_argument("--extracted", type=Path, required=True)
    c.add_argument("--independent", type=Path, required=True)
    c.add_argument("--repeat", type=Path, default=None,
                   help="second independent copy of some runs (run-to-run noise control)")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.src, a.out, [int(x) for x in a.budgets.split(",")])
    else:
        compare(a.extracted, a.independent, a.repeat)


if __name__ == "__main__":
    main()
