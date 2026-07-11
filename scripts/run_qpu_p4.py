"""QPU campaign — tranche P4: enhanced quantum neighborhoods, scaling n=50..500.

Fills the quantum rows of tab:rpd_scaling — the headline claim of the
windowed-QUBO article: QPU evaluation of full/windowed QUBOs at sizes
where the dense formulations stopped embedding.

Protocol follows the scaling table convention (classical baseline):
    ILS only, t_max = 5000 ms, 10 instances x 5 seeds per cell,
    m = 10 for n <= 200 (tai50_10, tai100_10, tai200_10),
    m = 20 for n = 500 (tai500_20).

Cells, in run order (cheap/high-value first, embedding-risky last):
    fibonacci_enhanced  n=50, adjacent_enhanced n=50,
    fibonacci_enhanced n=100, adjacent_enhanced n=100,
    fibonacci_enhanced n=200, fibonacci_enhanced n=500,
    adjacent_enhanced  n=200   <- dense K_199, at the ~190-variable
                                  capacity bound; embedding failures
                                  are expected and recorded as data.

Budget estimate (32 ms access/submission, 1 submission/iteration,
2-3 iterations per run at tl=5000): ~25 s of the 40 s tranche cap.
The self-metered guard (quantum.qpu_budget_s) cuts cleanly; failed
runs retry on --resume.

Usage:
    .venv311/bin/python3 scripts/run_qpu_p4.py
    .venv311/bin/python3 scripts/run_qpu_p4.py --limit 4        # micro-test
    .venv311/bin/python3 scripts/run_qpu_p4.py --resume results/experiments/<timestamp>

Requires DWAVE_API_TOKEN in the environment (.env).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

# (neighborhood, instance_file) in publication-priority order
CELLS = (
    ("quantum_fibonacci_enhanced", "data/tai50_10.txt"),
    ("quantum_adjacent_enhanced", "data/tai50_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai100_10.txt"),
    ("quantum_adjacent_enhanced", "data/tai100_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai200_10.txt"),
    ("quantum_fibonacci_enhanced", "data/tai500_20.txt"),
    ("quantum_adjacent_enhanced", "data/tai200_10.txt"),
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
