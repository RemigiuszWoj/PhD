"""Classical campaign of the crossing paper (articles/crossings/PLAN.md, option OPTIMUM by default).

Same runner, ILS/SA and constants as the previous papers and the k-bonacci campaign (config.yaml;
identity start; seeds 0-4; block filter of the previous papers). Runs are sequential, one process.
Every run uses t_max = 10 s; the budgets 0.1-5 s are read off its move log with
scripts/extract_budgets.py (validated in the k-bonacci campaign), run automatically after each stage.

Stages (--stage all = main, anatomy, ablation; each one resumes where it stopped):
  main      12 Taillard files x 11 modes: Dynasearch, Motzkin (c0), c1, P1, involutions, each plain and
            _safe, plus cross_swap (best single swap). cross_dyn is k-bonacci's kbonacci_kn (same
            candidates and DP); cross_dyn_safe answers decision D12 of the plan.
  anatomy   E1: states sampled from 10 s ILS runs (cross_c0, cross_c0_safe, cross_inv_safe), analysed by
            scripts/crossing_anatomy.py (predicted vs real gain per family, crossings, pair additivity
            by geometry, QUBO couplers, second-order model). n = 500 is skipped (one-move regime).
  ablation  exact critical-path filter for cross_c0_safe and cross_inv_safe.
  extra     option MAXIMUM only: prefix safeguard (cross_c0_prefix, cross_inv_prefix).

Usage (Mac, repo root):
    python scripts/run_crossing_campaign.py --dry-run
    python scripts/run_crossing_campaign.py --stage main              # ILS only (default)
    python scripts/run_crossing_campaign.py --stage main --algos sa   # later, on request: add SA (resumes)
    python scripts/run_crossing_campaign.py --stage main --files tai20_5 --limit 3   # smoke test
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
ALGOS = ["ils", "sa"]
LONG = 10000
OUT = ROOT / "results" / "crossings"

FAMILIES = ["dyn", "c0", "c1", "pa1", "inv"]
MAIN_MODES = [f"cross_{f}{r}" for f in FAMILIES for r in ("", "_safe")] + ["cross_swap"]
STAGES = {
    "main": {"files": TAILLARD, "modes": MAIN_MODES},
    "ablation": {"files": TAILLARD, "modes": ["cross_c0_safe_cp", "cross_inv_safe_cp"]},
    "extra": {"files": TAILLARD, "modes": ["cross_c0_prefix", "cross_inv_prefix"]},
}
ORDER = ["main", "anatomy", "ablation"]          # "extra" only on explicit request (option MAXIMUM)
# Wall-clock overrun of runs whose first evaluation outlasts the budget: all cross_* modes evaluate
# every pair, like Dynasearch / k = n (n = 500: ~36 s per evaluation on the M4 Pro).
OVERRUN_S = {500: 30.0, 200: 1.0}


def configs(files, modes, algos):
    out = []
    for f in files:
        for mode in modes:
            for algo in algos:
                for i in INSTANCES:
                    for s in SEEDS:
                        out.append(RunConfig(algorithm=algo, neighborhood=mode, instance_file=f"data/{f}.txt",
                                             instance_number=i, seed=s, time_limit_ms=LONG,
                                             tabu_tenure=10 if algo == "ils" else None))
    return out


def n_of(instance_file: str) -> int:
    return int(Path(instance_file).stem.split("_")[0][3:])


def est_hours(cfgs) -> float:
    return sum(c.time_limit_ms / 1000 + OVERRUN_S.get(n_of(c.instance_file), 0.0) + 0.05 for c in cfgs) / 3600


def run_stage(name: str, files, algos, dry: bool, limit: int, modes_filter) -> None:
    if name == "anatomy":
        cmd = [sys.executable, "scripts/crossing_anatomy.py", "--files", ",".join(f for f in files if f != "tai500_20")]
        if dry:
            cmd.append("--dry-run")
        if limit:
            cmd += ["--limit", str(limit)]
        subprocess.run(cmd, check=True)
        return
    spec = STAGES[name]
    modes = [m for m in spec["modes"] if not modes_filter or m in modes_filter]
    fl = [f for f in spec["files"] if f in files]
    cfgs = configs(fl, modes, algos)
    res_dir = OUT / name
    res_dir.mkdir(parents=True, exist_ok=True)
    if not dry:
        write_meta(res_dir, name, algos, modes)
    runner = ExperimentRunner(base_results_dir=str(OUT), quantum_config={}, resume_dir=str(res_dir))
    todo = [c for c in cfgs if not runner._already_done(c)]
    if limit:
        todo = todo[:limit]
    print(f"=== {name}: {len(cfgs)} runs ({len(modes)} modes), {len(todo)} to do, ~{est_hours(todo):.1f} h")
    if dry:
        return
    if todo:
        runner.run(todo)
    subprocess.run([sys.executable, "scripts/extract_budgets.py", "extract", "--src", str(res_dir),
                    "--out", str(OUT / f"{name}_budgets")], check=True)
    print_summary(res_dir)


def write_meta(res_dir: Path, stage: str, algos, modes) -> None:
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
    try:
        import networkx
        nx_version = networkx.__version__
    except ImportError:
        nx_version = None
    meta = {
        "stage": stage, "algos": list(algos), "modes": list(modes),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0], "networkx": nx_version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "iterated_local_search": cfg.get("iterated_local_search"),
        "simulated_annealing": cfg.get("simulated_annealing"),
        "seeds": SEEDS, "instances": INSTANCES,
    }
    with open(res_dir / "campaign_meta.jsonl", "a") as fh:
        fh.write(json.dumps(meta) + "\n")


def print_summary(res_dir: Path) -> None:
    """Moves per run and, for cross_* modes, how the moves look (swaps, crossings, safeguard use)."""
    import json
    import statistics
    from collections import defaultdict

    rows = []
    for p in res_dir.rglob("result.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"[summary] CORRUPT (delete this directory and resume): {p.parent}")
    if not rows:
        return
    it, ns = defaultdict(list), defaultdict(list)
    for r in rows:
        c = r["config"]
        key = (c["algorithm"], Path(c["instance_file"]).stem, c["neighborhood"])
        it[key].append(r.get("iterations") or 0)
        if r.get("neighborhood_stats"):
            ns[key].append(r["neighborhood_stats"])
    modes = [m for m in MAIN_MODES + STAGES["ablation"]["modes"] + STAGES["extra"]["modes"]
             if any(k[2] == m for k in it)]
    short = {m: m.replace("cross_", "").replace("_safe", "-s").replace("_prefix", "-p").replace("_cp", "/cp")
             for m in modes}

    def table(title, cell):
        print(f"\n{title} ({res_dir.name}):")
        for algo in sorted({k[0] for k in it}):
            print(f"  {algo:<4}{'file':<10}" + "".join(f"{short[m]:>9}" for m in modes))
            for f in [x for x in TAILLARD if any(k[1] == x and k[0] == algo for k in it)]:
                print(f"      {f:<10}" + "".join(cell(algo, f, m) for m in modes))

    def moves(algo, f, m):
        v = it.get((algo, f, m))
        return f"{statistics.mean(v):9.0f}" if v else f"{'-':>9}"

    def ratio(num, den):
        def cell(algo, f, m):
            v = ns.get((algo, f, m))
            if not v:
                return f"{'-':>9}"
            a, b = sum(s[num] for s in v), sum(s[den] for s in v)
            return f"{a / b:9.2f}" if b else f"{'-':>9}"
        return cell

    table("Mean number of moves (t_max = 10 s)", moves)
    table("Swaps per move", ratio("swaps", "moves"))
    table("Crossings per move", ratio("crossings", "moves"))
    table("Safeguard share (moves where the single swap or a prefix replaced the compound move)",
          ratio("fallback", "moves"))
    table("Share of moves with the TOP_K cap binding", ratio("cap_top", "moves"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", help="all | main | anatomy | ablation | extra (comma list allowed)")
    ap.add_argument("--files", default=",".join(TAILLARD), help="subset of Taillard files")
    ap.add_argument("--modes", default="", help="subset of the stage's modes, e.g. to skip cross_dyn")
    ap.add_argument("--algos", default="ils", help="ils (default) | sa | ils,sa")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="at most N runs per stage (smoke test)")
    a = ap.parse_args()
    stages = ORDER if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    files = [f.strip() for f in a.files.split(",")]
    algos = [x.strip() for x in a.algos.split(",")]
    modes_filter = {m.strip() for m in a.modes.split(",") if m.strip()}
    if any(x not in ALGOS for x in algos):
        ap.error(f"--algos must be from {ALGOS}")
    for s in stages:
        if s not in STAGES and s != "anatomy":
            ap.error(f"unknown stage {s}")
    if not a.dry_run:
        try:
            import networkx  # noqa: F401
        except ImportError:
            sys.exit("networkx is missing: pip install 'networkx>=3.2' (cross_inv needs it)")
    for s in stages:
        run_stage(s, files, algos, a.dry_run, a.limit, modes_filter)


if __name__ == "__main__":
    main()
