"""Classical ablation — BackTrackJumpList (mushroom elite pool) vs uniform
random restart in ILS diversification.

Backs the article's "BackTrackJumpList contribution" paragraph with data.
Both arms run fresh under the same code version, so pairs are exact:
    2 arms x 4 classical neighborhoods x 3 machine counts (tai20_5/10/20)
    x 3 time budgets (1000, 5000, 10000 ms) x 10 instances x 5 seeds
    = 3600 runs, ILS only, no QPU involved.

Blocks are ordered (m, tl, arm) so that each (m, tl) pair completes with
both arms before the next starts — partial completion still yields fully
paired Wilcoxon samples.

Usage:
    .venv311/bin/python3 scripts/run_btjl_ablation.py
    .venv311/bin/python3 scripts/run_btjl_ablation.py --resume results/experiments/<timestamp>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

INSTANCE_FILES = ("data/tai20_5.txt", "data/tai20_10.txt", "data/tai20_20.txt")
NEIGHBORHOODS = ("adjacent", "fibonacci", "dynasearch", "motzkin")
TIME_LIMITS_MS = (1000, 5000, 10000)
ARMS = ("mushroom", "random_restart")
INSTANCES = range(10)
SEEDS = range(5)


def build_plan() -> list[RunConfig]:
    plan: list[RunConfig] = []
    for inst_file in INSTANCE_FILES:
        for tl in TIME_LIMITS_MS:
            for arm in ARMS:
                for neigh in NEIGHBORHOODS:
                    for inst in INSTANCES:
                        for seed in SEEDS:
                            plan.append(
                                RunConfig(
                                    algorithm="ils",
                                    neighborhood=neigh,
                                    instance_file=inst_file,
                                    instance_number=inst,
                                    seed=seed,
                                    time_limit_ms=tl,
                                    tabu_tenure=10,
                                    diversification=arm,
                                )
                            )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="existing timestamp dir to continue")
    ap.add_argument("--limit", type=int, default=None, help="run only first N runs")
    args = ap.parse_args()

    runner = ExperimentRunner(resume_dir=args.resume)
    print(f"[BTJL] results dir: {runner.timestamp_dir}")
    plan = build_plan()
    if args.limit is not None:
        plan = plan[: args.limit]
    print(f"[BTJL] planned runs: {len(plan)}")
    runner.run(plan)
    print(f"[BTJL] DONE. Results dir: {runner.timestamp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
