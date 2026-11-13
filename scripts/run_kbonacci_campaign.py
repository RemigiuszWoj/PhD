"""Classical campaign of the k-bonacci paper (option MAXIMUM, PLAN_v2_archiwum.md A.4).

Same runner, ILS/SA and constants as the previous papers (config.yaml:
iterated_local_search / simulated_annealing; identity start; seeds 0-4; block
filter of the previous papers). Runs are sequential, one process, as before.
Every run uses t_max = 10 s; the budgets 0.1-5 s are read off its move log
(scripts/extract_budgets.py), which is run automatically after each stage.

Stages (run in this order by --stage all; each one resumes where it stopped):
  validate  tai20_5 + tai50_10, k in {3, 8}, all six budgets as INDEPENDENT runs,
            plus a repeat of the 0.1-5 s runs (run-to-run noise control); prints
            VALIDATION GATE: PASS/FAIL and stops the campaign on FAIL.
  replicate tai20_5 + tai50_10, OLD neighborhoods fibonacci and dynasearch, 10 s:
            compared afterwards with the old campaign (same seeds) to confirm that
            the machine/Python state still matches the earlier runs.
  main      12 Taillard files, k in {2,3,4,5,6,8,12,16,n}.
  insert    12 Taillard files, Insert neighborhood (reference row).
  ablation  12 Taillard files, k in {3,8} with the exact critical-path filter.
  relative  12 Taillard files, k = ceil(alpha n) for alpha = 10%, ..., 90% (values not in `main`).
  ablation_all  12 Taillard files, exact filter for the remaining k of `main`.
  large     n = 200, 500, exact filter, k in {4,6,8,12,16,24,32} + Insert, 60 s (budgets to 60 s).

Usage (Mac, repo root):
    python scripts/run_kbonacci_campaign.py --dry-run
    python scripts/run_kbonacci_campaign.py --stage all              # ILS only (default)
    python scripts/run_kbonacci_campaign.py --stage all --algos sa   # later: add SA (resumes)
    python scripts/run_kbonacci_campaign.py --stage main --files tai20_5,tai20_10
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
sys.setrecursionlimit(1_000_000)  # dynasearch_full (replicate) uses a recursive DP (depth ~ #candidates)

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

TAILLARD = [  # ascending n: errors show up on the fast classes first
    "tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20",
    "tai100_5", "tai100_10", "tai100_20", "tai200_10", "tai200_20", "tai500_20",
]
INSTANCES = list(range(10))
SEEDS = list(range(5))
ALGOS = ["ils", "sa"]  # available; default run = ILS only (--algos), SA added later if worthwhile
ALL_BUDGETS = [100, 500, 1000, 2000, 5000, 10000]
LONG = 10000
OUT = ROOT / "results" / "kbonacci"

K_MAIN = ["kbonacci_k2", "kbonacci_k3", "kbonacci_k4", "kbonacci_k5", "kbonacci_k6",
          "kbonacci_k8", "kbonacci_k12", "kbonacci_k16", "kbonacci_kn"]
STAGES = {
    "validate": {"files": ["tai20_5", "tai50_10"], "modes": ["kbonacci_k3", "kbonacci_k8"],
                 "budgets": ALL_BUDGETS, "repeat_budgets": [100, 500, 1000, 2000, 5000]},
    # Replication control: two OLD neighborhoods, re-run now with the same seeds, to be
    # compared with the old campaign (same Mac? same Python/package speed?).
    "replicate": {"files": ["tai20_5", "tai50_10"], "modes": ["fibonacci", "dynasearch"], "budgets": [LONG]},
    "main": {"files": TAILLARD, "modes": K_MAIN, "budgets": [LONG]},
    "insert": {"files": TAILLARD, "modes": ["insert"], "budgets": [LONG]},
    "ablation": {"files": TAILLARD, "modes": ["kbonacci_k3_cp", "kbonacci_k8_cp"], "budgets": [LONG]},
}
# Relative k: k = ceil(alpha * n) for alpha = 10%, 20%, ..., 90%, so that every n has points
# between the fixed small k of `main` and k = n. Values already in `main` are skipped.
REL_ALPHA = [a / 10 for a in range(1, 10)]


def relative_modes(f: str) -> list:
    n = int(f.split("_")[0][3:])
    ks = sorted({max(2, math.ceil(a * n)) for a in REL_ALPHA})
    return [f"kbonacci_k{k}" for k in ks if f"kbonacci_k{k}" not in K_MAIN and k < n]


STAGES["relative"] = {"files": TAILLARD, "modes": relative_modes, "budgets": [LONG]}
# Exact critical-path filter for the remaining k of `main` (k = 3 and 8 are in `ablation`).
STAGES["ablation_all"] = {"files": TAILLARD, "budgets": [LONG],
                          "modes": [f"{m}_cp" for m in K_MAIN if m not in ("kbonacci_k3", "kbonacci_k8")]}
# Large n, long budget: does a small k win at n >= 200 by nature or only because 10 s is short?
# Exact filter only; budgets up to 60 s are read off the move log of the 60 s runs.
STAGES["large"] = {"files": ["tai200_10", "tai200_20", "tai500_20"], "budgets": [60000],
                   "modes": [f"kbonacci_k{k}_cp" for k in (4, 6, 8, 12, 16, 24, 32)] + ["insert"],
                   "extract_budgets": [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 60000]}
ORDER = ["validate", "replicate", "main", "insert", "ablation", "relative", "ablation_all", "large"]
# Rough wall-clock overrun for runs whose first move outlasts the budget
# (k = n at n = 500: one Dynasearch-sized evaluation takes ~36 s in Python).
OVERRUN_S = {("kbonacci_kn", 500): 30.0}


def overrun_s(mode: str, n: int) -> float:
    if (mode, n) in OVERRUN_S:
        return OVERRUN_S[(mode, n)]
    k = mode[len("kbonacci_k"):] if mode.startswith("kbonacci_k") else ""
    # relative k at n = 500: the first move scales roughly with k / n of the k = n case
    return 30.0 * int(k) / n if n == 500 and k.isdigit() and int(k) > 16 else 0.0


def configs(files, modes, budgets, algos):
    out = []
    for f in files:
        for mode in (modes(f) if callable(modes) else modes):
            for algo in algos:
                for t in budgets:
                    for i in INSTANCES:
                        for s in SEEDS:
                            out.append(RunConfig(algorithm=algo, neighborhood=mode,
                                                 instance_file=f"data/{f}.txt", instance_number=i,
                                                 seed=s, time_limit_ms=t,
                                                 tabu_tenure=10 if algo == "ils" else None))
    return out


def est_hours(cfgs) -> float:
    sec = 0.0
    for c in cfgs:
        n = int(Path(c.instance_file).stem.split("_")[0][3:])
        sec += c.time_limit_ms / 1000 + overrun_s(c.neighborhood, n) + 0.05
    return sec / 3600


def run_stage(name: str, files, algos, dry: bool, limit: int) -> None:
    spec = STAGES[name]
    fl = [f for f in spec["files"] if f in files]
    jobs = [(OUT / name, configs(fl, spec["modes"], spec["budgets"], algos))]
    if spec.get("repeat_budgets"):
        jobs.append((OUT / f"{name}_repeat", configs(fl, spec["modes"], spec["repeat_budgets"], algos)))
    for res_dir, cfgs in jobs:
        res_dir.mkdir(parents=True, exist_ok=True)
        if not dry:
            write_meta(res_dir, name, algos)
        runner = ExperimentRunner(base_results_dir=str(OUT), quantum_config={}, resume_dir=str(res_dir))
        todo = [c for c in cfgs if not runner._already_done(c)]
        if limit:
            todo = todo[:limit]
        print(f"=== {res_dir.name}: {len(cfgs)} runs, {len(todo)} to do, ~{est_hours(todo):.1f} h")
        if not dry and todo:
            runner.run(todo)
    if dry:
        return
    # budgets from the 10 s runs (validate: from its own 10 s runs, for the comparison)
    src, out = OUT / name, OUT / f"{name}_budgets"
    extra = ["--budgets", ",".join(map(str, spec["extract_budgets"]))] if spec.get("extract_budgets") else []
    subprocess.run([sys.executable, "scripts/extract_budgets.py", "extract", "--src", str(src),
                    "--out", str(out), *extra], check=True)
    print_moves(src)
    if name == "validate":
        gate = subprocess.run([sys.executable, "scripts/extract_budgets.py", "compare",
                               "--extracted", str(out), "--independent", str(src),
                               "--repeat", str(OUT / "validate_repeat")])
        if gate.returncode != 0:
            print("VALIDATION GATE FAILED -- stopping here; no further stage is started.")
            sys.exit(2)


def write_meta(res_dir: Path, stage: str, algos) -> None:
    """Provenance of every (re)start of a stage: commit, machine, Python, protocol constants."""
    import datetime
    import json
    import platform

    import yaml

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    with open("config.yaml") as fh:
        cfg = yaml.safe_load(fh) or {}
    meta = {
        "stage": stage, "algos": list(algos),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "iterated_local_search": cfg.get("iterated_local_search"),
        "simulated_annealing": cfg.get("simulated_annealing"),
        "seeds": SEEDS, "instances": INSTANCES,
    }
    log = res_dir / "campaign_meta.jsonl"
    with open(log, "a") as fh:
        fh.write(json.dumps(meta) + "\n")


def print_moves(res_dir: Path) -> None:
    """How many moves (metaheuristic iterations) each neighborhood makes in the full budget."""
    import json
    import statistics
    from collections import defaultdict

    rows = []
    for p in res_dir.rglob("result.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"[moves] CORRUPT (delete this directory and resume): {p.parent}")
    if not rows:
        return
    t_long = max(r["config"]["time_limit_ms"] for r in rows)
    it = defaultdict(list)
    for r in rows:
        c = r["config"]
        if c["time_limit_ms"] == t_long:
            it[(c["algorithm"], Path(c["instance_file"]).stem, c["neighborhood"])].append(r.get("iterations") or 0)
    modes = sorted({k[2] for k in it}, key=lambda m: (len(m), m))
    print(f"\nMean number of moves in {res_dir.name} (t_max = {t_long} ms):")
    for algo in sorted({k[0] for k in it}):
        print(f"  {algo:<4}{'file':<10}" + "".join(f"{m.replace('kbonacci_', ''):>9}" for m in modes))
        for f in [x for x in TAILLARD if any(k[1] == x and k[0] == algo for k in it)]:
            cells = [f"{statistics.mean(it[(algo, f, m)]):9.0f}" if it.get((algo, f, m)) else f"{'-':>9}"
                     for m in modes]
            print(f"      {f:<10}" + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | " + " | ".join(ORDER) + " (comma list allowed)")
    ap.add_argument("--files", default=",".join(TAILLARD), help="subset of Taillard files")
    ap.add_argument("--algos", default="ils", help="ils (default) | sa | ils,sa")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="at most N runs per stage (smoke test)")
    a = ap.parse_args()
    stages = ORDER if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    files = [f.strip() for f in a.files.split(",")]
    algos = [x.strip() for x in a.algos.split(",")]
    if any(x not in ALGOS for x in algos):
        ap.error(f"--algos must be from {ALGOS}")
    for s in stages:
        if s not in STAGES:
            ap.error(f"unknown stage {s}")
        run_stage(s, files, algos, a.dry_run, a.limit)


if __name__ == "__main__":
    main()
