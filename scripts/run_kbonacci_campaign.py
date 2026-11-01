"""Classical campaign of the k-bonacci paper (option MAXIMUM, PLAN.md A.4).

Same runner, ILS/SA and constants as the previous papers (config.yaml:
iterated_local_search / simulated_annealing; identity start; seeds 0-4; block
filter of the previous papers). Runs are sequential, one process, as before.
Every run uses t_max = 10 s; the budgets 0.1-5 s are read off its move log
(scripts/extract_budgets.py), which is run automatically after each stage.

Stages (run in this order by --stage all; each one resumes where it stopped):
  validate  tai20_5 + tai50_10, k in {3, 8}, all six budgets as INDEPENDENT runs,
            plus a repeat of the 1 s and 5 s runs (run-to-run noise control).
  main      12 Taillard files, k in {2,3,4,5,6,8,12,16,n}.
  insert    12 Taillard files, Insert neighborhood (reference row).
  ablation  12 Taillard files, k in {3,8} with the exact critical-path filter.

Usage (Mac, repo root):
    python scripts/run_kbonacci_campaign.py --dry-run
    python scripts/run_kbonacci_campaign.py --stage all              # ILS only (default)
    python scripts/run_kbonacci_campaign.py --stage all --algos sa   # later: add SA (resumes)
    python scripts/run_kbonacci_campaign.py --stage main --files tai20_5,tai20_10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

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
                 "budgets": ALL_BUDGETS, "repeat_budgets": [1000, 5000]},
    "main": {"files": TAILLARD, "modes": K_MAIN, "budgets": [LONG]},
    "insert": {"files": TAILLARD, "modes": ["insert"], "budgets": [LONG]},
    "ablation": {"files": TAILLARD, "modes": ["kbonacci_k3_cp", "kbonacci_k8_cp"], "budgets": [LONG]},
}
ORDER = ["validate", "main", "insert", "ablation"]
# Rough wall-clock overrun for runs whose first move outlasts the budget
# (k = n at n = 500: one Dynasearch-sized evaluation takes ~36 s in Python).
OVERRUN_S = {("kbonacci_kn", 500): 30.0}


def configs(files, modes, budgets, algos):
    out = []
    for f in files:
        for mode in modes:
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
        sec += c.time_limit_ms / 1000 + OVERRUN_S.get((c.neighborhood, n), 0.0) + 0.05
    return sec / 3600


def run_stage(name: str, files, algos, dry: bool, limit: int) -> None:
    spec = STAGES[name]
    fl = [f for f in spec["files"] if f in files]
    jobs = [(OUT / name, configs(fl, spec["modes"], spec["budgets"], algos))]
    if spec.get("repeat_budgets"):
        jobs.append((OUT / f"{name}_repeat", configs(fl, spec["modes"], spec["repeat_budgets"], algos)))
    for res_dir, cfgs in jobs:
        res_dir.mkdir(parents=True, exist_ok=True)
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
    subprocess.run([sys.executable, "scripts/extract_budgets.py", "extract", "--src", str(src),
                    "--out", str(out)], check=True)
    print_moves(src)
    if name == "validate":
        subprocess.run([sys.executable, "scripts/extract_budgets.py", "compare",
                        "--extracted", str(out), "--independent", str(src),
                        "--repeat", str(OUT / "validate_repeat")], check=True)


def print_moves(res_dir: Path) -> None:
    """How many moves (metaheuristic iterations) each neighborhood makes in the full budget."""
    import json
    import statistics
    from collections import defaultdict

    it = defaultdict(list)
    for p in res_dir.rglob("result.json"):
        r = json.loads(p.read_text())
        c = r["config"]
        it[(c["algorithm"], Path(c["instance_file"]).stem, c["neighborhood"])].append(r.get("iterations") or 0)
    if not it:
        return
    modes = sorted({k[2] for k in it}, key=lambda m: (len(m), m))
    print(f"\nMean number of moves in {res_dir.name} (t_max of the stage):")
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
