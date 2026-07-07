"""QPU campaign — tranche P2: quantum_qubo_enhanced neighborhoods, n=20.

Fills the "Quantum QUBO Enhanced" rows of tab:rpd_n20 — the controlled
measurement of the delta-filter effect promised in the article prose.
Protocol identical to P1 (classical baseline / PAN article):
    tai20_5 × 10 instances × 5 seeds × {ILS, SA} × 6 time limits
    × 4 enhanced neighborhoods = 2400 runs.

Time limits ordered by publication priority: tl=1000 ms first (the
tab:rpd_n20 cells), then 5000 (convergence point), then the rest.
The self-metered budget (quantum.qpu_budget_s) cuts the batch cleanly;
budget-failed runs retry on --resume.

Expect windowed dynasearch/motzkin to be SLOW per call (minorminer on
~170-variable windows) — wall time, not QPU budget.

Usage:
    .venv311/bin/python3 scripts/run_qpu_p2.py
    .venv311/bin/python3 scripts/run_qpu_p2.py --resume results/experiments/<timestamp>

Requires DWAVE_API_TOKEN in the environment (.env).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

INSTANCE_FILE = "data/tai20_5.txt"
NEIGHBORHOODS = (
    "quantum_adjacent_enhanced",
    "quantum_fibonacci_enhanced",
    "quantum_dynasearch_enhanced",
    "quantum_motzkin_enhanced",
)
ALGOS = ("ils", "sa")
TIME_LIMITS_MS = (1000, 5000, 500, 100, 2000, 10000)
INSTANCES = range(10)
SEEDS = range(5)


def build_plan() -> list[RunConfig]:
    plan: list[RunConfig] = []
    for tl in TIME_LIMITS_MS:
        for neigh in NEIGHBORHOODS:
            for algo in ALGOS:
                for inst in INSTANCES:
                    for seed in SEEDS:
                        plan.append(
                            RunConfig(
                                algorithm=algo,
                                neighborhood=neigh,
                                instance_file=INSTANCE_FILE,
                                instance_number=inst,
                                seed=seed,
                                time_limit_ms=tl,
                                tabu_tenure=10 if algo == "ils" else None,
                            )
                        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="existing timestamp dir to continue")
    args = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qconf = dict(cfg.get("quantum") or {})
    qconf["backend"] = "dwave"  # explicit opt-in for this campaign script

    print(f"[P2] backend=dwave, num_reads_enhanced={qconf.get('num_reads_enhanced')}, "
          f"qpu_budget_s={qconf.get('qpu_budget_s')}, resume={args.resume or 'no'}")

    runner = ExperimentRunner(quantum_config=qconf, resume_dir=args.resume)
    print(f"[P2] results dir: {runner.timestamp_dir}")
    plan = build_plan()
    print(f"[P2] planned runs: {len(plan)}")
    runner.run(plan)
    print(f"[P2] DONE. Results dir: {runner.timestamp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
