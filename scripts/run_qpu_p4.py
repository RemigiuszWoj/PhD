"""QPU campaign — tranche P4+: fill ALL empty cells of tab:rpd_scaling.

The scaling table (windowed-QUBO article) is ILS, t_max = 5000 ms,
m = 10 for n <= 200, m = 20 for n = 500. Every "--" cell is measured
here; n/a cells stay n/a (documented tractability bounds: dense QUBOs
stop embedding at n >= 50, windowed dyn/motz exceed the wall-time
bound at n >= 100).

Plan, 17 cells x 50 runs (10 instances x 5 seeds) = 850 runs:

  Phase A — n=20 column at m=10 (tai20_10), all 8 quantum rows.
            Cheap cells first; original dynasearch last in phase
            (minorminer on ~190 dense variables, ~2 min wall/call,
            expect ~4% embedding failures as at m=5).
  Phase B — scaling, fibonacci/adjacent enhanced:
            n=50, 100, 200 (m=10) and fibonacci n=500 (m=20).
            Adjacent n=200 last: dense K_199 sits at the ~190-variable
            embedding capacity bound; failures are recorded as data.
  Phase C — windowed dynasearch/motzkin enhanced at n=50 (~5 windows
            per call, minorminer grind per window -> wall-heavy,
            budget-cheap).

Budget estimate (32 ms access/submission): ~70 s total, i.e. two
40 s tranches. The self-metered guard (quantum.qpu_budget_s) cuts
each tranche cleanly; budget-failed runs retry on --resume.

Usage:
    .venv311/bin/python3 scripts/run_qpu_p4.py
    .venv311/bin/python3 scripts/run_qpu_p4.py --limit 2         # micro-test
    .venv311/bin/python3 scripts/run_qpu_p4.py --resume results/experiments/<timestamp>

Requires DWAVE_API_TOKEN in the environment (.env).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

# (neighborhood, instance_file) in execution order
CELLS = (
    # Phase A: n=20 column, m=10
    ("quantum_adjacent", "data/tai20_10.txt"),
    ("quantum_fibonacci", "data/tai20_10.txt"),
    ("quantum_adjacent_enhanced", "data/tai20_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai20_10.txt"),
    ("quantum_motzkin", "data/tai20_10.txt"),
    ("quantum_motzkin_enhanced", "data/tai20_10.txt"),
    ("quantum_dynasearch_enhanced", "data/tai20_10.txt"),
    ("quantum_dynasearch", "data/tai20_10.txt"),
    # Phase B: fibonacci/adjacent enhanced scaling
    ("quantum_fibonacci_enhanced", "data/tai50_10.txt"),
    ("quantum_adjacent_enhanced", "data/tai50_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai100_10.txt"),
    ("quantum_adjacent_enhanced", "data/tai100_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai200_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai500_20.txt"),
    ("quantum_adjacent_enhanced", "data/tai200_10.txt"),
    # Phase C: windowed neighborhoods at n=50 (wall-heavy)
    ("quantum_motzkin_enhanced", "data/tai50_10.txt"),
    ("quantum_dynasearch_enhanced", "data/tai50_10.txt"),
)
TIME_LIMIT_MS = 5000
INSTANCES = range(10)
SEEDS = range(5)


def build_plan() -> list[RunConfig]:
    plan: list[RunConfig] = []
    for neigh, inst_file in CELLS:
        for inst in INSTANCES:
            for seed in SEEDS:
                plan.append(
                    RunConfig(
                        algorithm="ils",
                        neighborhood=neigh,
                        instance_file=inst_file,
                        instance_number=inst,
                        seed=seed,
                        time_limit_ms=TIME_LIMIT_MS,
                        tabu_tenure=10,
                    )
                )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="existing timestamp dir to continue")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N runs of the plan (micro-test)")
    args = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qconf = dict(cfg.get("quantum") or {})
    qconf["backend"] = "dwave"  # explicit opt-in for this campaign script

    print(f"[P4] backend=dwave, num_reads_enhanced={qconf.get('num_reads_enhanced')}, "
          f"qpu_budget_s={qconf.get('qpu_budget_s')}, resume={args.resume or 'no'}")

    runner = ExperimentRunner(quantum_config=qconf, resume_dir=args.resume)
    print(f"[P4] results dir: {runner.timestamp_dir}")
    plan = build_plan()
    if args.limit is not None:
        plan = plan[: args.limit]
    print(f"[P4] planned runs: {len(plan)}")
    runner.run(plan)
    print(f"[P4] DONE. Results dir: {runner.timestamp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
