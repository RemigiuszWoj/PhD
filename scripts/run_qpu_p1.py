"""QPU campaign — tranche P1: original quantum_qubo neighborhoods, n=20.

Protocol mirrors the classical baseline (PAN article):
    tai20_5 × 10 instances × 5 seeds × {ILS, SA} × 6 time limits
    × 4 neighborhoods (quantum_adjacent/fibonacci/dynasearch/motzkin)
    = 2400 runs total.

Time limits are ordered by PUBLICATION PRIORITY, not magnitude:
tl=1000 ms first (fills all 16 quantum cells of tab:rpd_n20), then the
remaining budgets for gap-vs-time curves. The self-metered QPU budget
(quantum.qpu_budget_s in config.yaml) cuts the batch cleanly when the
tranche allowance is spent; unfinished runs are recorded as failed.json
and are RETRIED automatically on resume.

Usage:
    # fresh tranche (creates new timestamp dir):
    .venv311/bin/python3 scripts/run_qpu_p1.py
    # continue after raising qpu_budget_s (next tranche):
    .venv311/bin/python3 scripts/run_qpu_p1.py --resume results/experiments/<timestamp>

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
    "quantum_adjacent",
    "quantum_fibonacci",
    "quantum_dynasearch",
    "quantum_motzkin",
)
ALGOS = ("ils", "sa")
# Publication priority: 1000 ms fills tab:rpd_n20; the rest builds curves.
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

    budget = qconf.get("qpu_budget_s")
    print(f"[P1] backend=dwave, num_reads={qconf.get('num_reads')}, "
          f"qpu_budget_s={budget}, resume={args.resume or 'no'}")

    runner = ExperimentRunner(quantum_config=qconf, resume_dir=args.resume)
    print(f"[P1] results dir: {runner.timestamp_dir}")
    plan = build_plan()
    print(f"[P1] planned runs: {len(plan)}")
    runner.run(plan)
    print(f"[P1] DONE. Results dir: {runner.timestamp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
