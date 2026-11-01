"""Job shop campaign (articles/jobshop/PLAN.md, options MIN / OPT / MAX).

Protocol = the PFSP papers: ILS (and later SA) with the same control flow and constants
(src/jobshop/algorithms.py), round-robin start (identity analog), seeds 0-4, one 10 s run
per configuration, budgets 0.1-5 s read off its move log (scripts/extract_budgets.py, run
automatically after each stage). Runs are sequential, one process.

Stages (each resumes where it stopped; run in this order):
  validate   ta01-10 + ta31-40, js_k3 and js_c6, all six budgets as INDEPENDENT runs,
             plus a repeat of the 1 s and 5 s runs (noise control)            [~1.4 h ILS]
  min        ta01-ta80, MIN modes (N5, Fib, k=3, Dyn, N6, c6)                 [~6.7 h ILS]
  opt        ta01-ta80, extra OPT modes (N1, k=4, N7, c7, bm3)                [~5.6 h ILS]
  continuum  generated instances, sizes 20x5, 20x10, 30x15, 50x10, rho in {0, .1, .25, .5, 1},
             10 instances, seeds 0-2, 5 s, modes N5, k=3, Dyn, N6, c6        [~4.2 h ILS]
  bridge     Taillard PFSP tai20_5, tai20_10, tai50_10 (inst. 0-9) read as job shops
             (identical routes, non-permutation schedules); needs data/tai*.txt [~1.3 h ILS]
  ablation   (MAX) ta01-10 + ta31-40: exact deltas and job tie-break           [~1.1 h ILS]
  classic    (MAX) ft/la/orb/abz/swv/yn, modes N5, k=3, N6, c6                 [~4.6 h ILS]

Usage (Mac, repo root, after scripts/fetch_jsplib.py and scripts/verify_taillard_jssp.py):
    python scripts/run_jobshop_campaign.py --dry-run
    python scripts/run_jobshop_campaign.py --stage validate
    python scripts/run_jobshop_campaign.py --stage min,opt,continuum,bridge
    python scripts/run_jobshop_campaign.py --stage min --algos sa        # later, on request
    python scripts/run_jobshop_campaign.py --stage min --classes 15x15 --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.jobshop.instance import JSPLIB_DIR, TA_CLASSES, class_of, ta_name  # noqa: E402
from src.jobshop.runner import ILS_DEFAULTS, SA_DEFAULTS, JSRunConfig, already_done, run_one  # noqa: E402

OUT = ROOT / "results" / "jobshop"
SEEDS = list(range(5))
ALL_BUDGETS = [100, 500, 1000, 2000, 5000, 10000]
LONG = 10000

MIN_MODES = ["js_n5", "js_fib", "js_k3", "js_kn", "js_n6", "js_c6"]
OPT_EXTRA = ["js_n1", "js_k4", "js_n7", "js_c7", "js_bm3"]
CONT_MODES = ["js_n5", "js_k3", "js_kn", "js_n6", "js_c6"]
CONT_SIZES = [(20, 5), (20, 10), (30, 15), (50, 10)]
CONT_RHOS = [0.0, 0.1, 0.25, 0.5, 1.0]
CLASSIC = ([f"ft{x}" for x in ("06", "10", "20")] + [f"la{i:02d}" for i in range(1, 41)]
           + [f"orb{i:02d}" for i in range(1, 11)] + [f"abz{i}" for i in range(5, 10)]
           + [f"swv{i:02d}" for i in range(1, 21)] + [f"yn{i}" for i in range(1, 5)])

TA_ALL = [ta_name(i) for c in TA_CLASSES.values() for i in c]
TA_VALID = [ta_name(i) for i in range(1, 11)] + [ta_name(i) for i in range(31, 41)]


def _cfgs(files, modes, budgets, algos, seeds=SEEDS, numbers=(0,)):
    out = []
    for f in files:
        for mode in modes:
            for algo in algos:
                for t in budgets:
                    for i in numbers:
                        for s in seeds:
                            out.append(JSRunConfig(algorithm=algo, neighborhood=mode, instance_file=f,
                                                   instance_number=i, seed=s, time_limit_ms=t,
                                                   tabu_tenure=10 if algo == "ils" else None))
    return out


def stage_jobs(name: str, algos, classes):
    ta = [f for f in TA_ALL if class_of(f) in classes]
    tav = [f for f in TA_VALID if class_of(f) in classes]
    if name == "validate":
        return [("validate", _cfgs(tav, ["js_k3", "js_c6"], ALL_BUDGETS, algos)),
                ("validate_repeat", _cfgs(tav, ["js_k3", "js_c6"], [1000, 5000], algos))]
    if name == "min":
        return [("min", _cfgs(ta, MIN_MODES, [LONG], algos))]
    if name == "opt":
        return [("opt", _cfgs(ta, OPT_EXTRA, [LONG], algos))]
    if name == "continuum":
        files = [f"gen:{n}x{m}:rho={r}:{k}" for (n, m) in CONT_SIZES for r in CONT_RHOS for k in range(10)]
        return [("continuum", _cfgs(files, CONT_MODES, [5000], algos, seeds=[0, 1, 2]))]
    if name == "bridge":
        files = [f"pfsp:{f}:{i}" for f in ("tai20_5", "tai20_10", "tai50_10") for i in range(10)]
        return [("bridge", _cfgs(files, ["js_n5", "js_k3", "js_c6"], [LONG], algos))]
    if name == "ablation":
        return [("ablation", _cfgs(tav, ["js_k3_exact", "js_c6_exact", "js_k3_job", "js_c6_job"], [LONG], algos))]
    if name == "classic":
        files = [f for f in CLASSIC if (JSPLIB_DIR / f).exists()]
        return [("classic", _cfgs(files, ["js_n5", "js_k3", "js_n6", "js_c6"], [LONG], algos))]
    raise ValueError(name)


ORDER = ["validate", "min", "opt", "continuum", "bridge", "ablation", "classic"]


def est_hours(cfgs) -> float:
    return sum(c.time_limit_ms / 1000 + 0.15 for c in cfgs) / 3600


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_meta(res_dir: Path, stage: str, algos) -> None:
    meta = {"stage": stage, "algos": list(algos), "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "ils": ILS_DEFAULTS, "sa": SA_DEFAULTS, "seeds": SEEDS}
    with open(res_dir / "campaign_meta.jsonl", "a") as fh:
        fh.write(json.dumps(meta) + "\n")


def print_moves(res_dir: Path) -> None:
    """Mean number of moves (loop passes) per class and mode at the stage's longest budget."""
    rows = [json.loads(p.read_text()) for p in res_dir.rglob("result.json")]
    if not rows:
        return
    t_long = max(r["config"]["time_limit_ms"] for r in rows)
    it = defaultdict(list)
    for r in rows:
        c = r["config"]
        if c["time_limit_ms"] != t_long:
            continue
        f = c["instance_file"]
        grp = class_of(f) or (f.split(":")[1] + (f":{f.split(':')[2]}" if f.startswith("gen:") else "")
                              if ":" in f else f)
        it[(c["algorithm"], grp, c["neighborhood"])].append(r.get("iterations") or 0)
    modes = sorted({k[2] for k in it})
    print(f"\nMean number of moves in {res_dir.name} (t_max = {t_long} ms):")
    for algo in sorted({k[0] for k in it}):
        print(f"  {algo:<4}{'group':<16}" + "".join(f"{m.replace('js_', ''):>9}" for m in modes))
        for g in sorted({k[1] for k in it if k[0] == algo}, key=lambda x: (len(x), x)):
            cells = [f"{statistics.mean(it[(algo, g, m)]):9.0f}" if it.get((algo, g, m)) else f"{'-':>9}" for m in modes]
            print(f"      {g:<16}" + "".join(cells))


def run_stage(name: str, algos, classes, dry: bool, limit: int) -> None:
    if name == "bridge" and not (ROOT / "data" / "tai20_5.txt").exists():
        print("=== bridge: SKIPPED, data/tai20_5.txt (Taillard PFSP) not found", flush=True)
        return
    for sub, cfgs in stage_jobs(name, algos, classes):
        res_dir = OUT / sub
        res_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in cfgs if not already_done(res_dir, c)]
        if limit:
            todo = todo[:limit]
        print(f"=== {sub}: {len(cfgs)} runs, {len(todo)} to do, ~{est_hours(todo):.1f} h", flush=True)
        if dry or not todo:
            continue
        write_meta(res_dir, sub, algos)
        t0 = time.time()
        for k, c in enumerate(todo, 1):
            d = run_one(c, res_dir)
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{sub}] {k}/{len(todo)}  last: {c.algorithm} {c.neighborhood} {c.instance_file} "
                      f"seed={c.seed} C={d['cmax_best']} it={d['iterations']}  "
                      f"elapsed {el / 3600:.2f} h, ETA {el / k * (len(todo) - k) / 3600:.2f} h", flush=True)
    if dry:
        return
    src, out = OUT / name, OUT / f"{name}_budgets"
    budgets = "100,500,1000,2000,5000" if name == "continuum" else "100,500,1000,2000,5000,10000"
    subprocess.run([sys.executable, "scripts/extract_budgets.py", "extract", "--src", str(src), "--out", str(out),
                    "--budgets", budgets], check=True)
    print_moves(src)
    if name == "validate":
        subprocess.run([sys.executable, "scripts/extract_budgets.py", "compare", "--extracted", str(out),
                        "--independent", str(src), "--repeat", str(OUT / "validate_repeat")], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="validate", help=f"comma list of {ORDER} or 'all' (= validate..bridge)")
    ap.add_argument("--algos", default="ils", help="ils, sa or ils,sa")
    ap.add_argument("--classes", default=",".join(TA_CLASSES), help="Taillard classes to include, e.g. 15x15,20x15")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many configurations per stage")
    a = ap.parse_args()
    stages = ORDER[:5] if a.stage == "all" else a.stage.split(",")
    algos = a.algos.split(",")
    classes = set(a.classes.split(","))
    if not (JSPLIB_DIR / "ta01").exists() and any(s not in ("continuum", "bridge") for s in stages):
        sys.exit("data/JSPLIB/instances missing: run scripts/fetch_jsplib.py and scripts/verify_taillard_jssp.py first")
    for s in stages:
        run_stage(s, algos, classes, a.dry_run, a.limit)
    if a.dry_run:
        tot = sum(est_hours(c) for s in stages for _, c in stage_jobs(s, algos, classes))
        print(f"=== total (stages {','.join(stages)}, algos {','.join(algos)}): ~{tot:.1f} h")


if __name__ == "__main__":
    main()
