#!/usr/bin/env python3
"""Run a chunk of FAZA 2d (n=500) for slow neighborhoods (dynasearch, motzkin).

Usage:
  .venv311/bin/python3 scripts/run_n500_chunk.py --neigh dynasearch --inst 0-2
  .venv311/bin/python3 scripts/run_n500_chunk.py --neigh motzkin    --inst 5,7

The runs land in results/experiments/20260529_065745/ (same dir as adj+fib 2d),
so resume_dir behavior keeps everything in one experiment directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Dynasearch DP recurses depth O(num_candidates) = O(n^2) ≈ 124k at n=500.
# Match the limit set in main.py.
sys.setrecursionlimit(2_000_000)

from src.experiments.runner import (
    ExperimentRunner,
    RunConfig,
    count_instances_in_file,
)


N500_FILE = "data/tai500_20.txt"
N500_RESULTS_DIR = "results/experiments/20260529_065745"
REPEATS = 5
DEFAULT_TIME_LIMITS_MS = [10000]  # at n=500 dyn/motz, all tls degenerate to 1 iter
ALGOS = ("ils", "sa")
TABU_TENURE = 10


def parse_inst_spec(spec: str, max_inst: int) -> list[int]:
    """Parse '0-2' or '0,2,3' or '5' into list of ints."""
    result: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-")
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(token))
    out = sorted(i for i in result if 0 <= i < max_inst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neigh", required=True, choices=["dynasearch", "motzkin"])
    ap.add_argument("--inst", required=True, help="e.g. '0-2' or '0,2,5'")
    ap.add_argument(
        "--tl",
        default=",".join(str(t) for t in DEFAULT_TIME_LIMITS_MS),
        help=f"comma-separated time limits in ms (default: {DEFAULT_TIME_LIMITS_MS})",
    )
    args = ap.parse_args()
    time_limits = [int(t) for t in args.tl.split(",")]

    n_inst = count_instances_in_file(N500_FILE)
    instances = parse_inst_spec(args.inst, n_inst)
    if not instances:
        print(f"No valid instances in '{args.inst}' (max={n_inst})", file=sys.stderr)
        sys.exit(1)

    # Build plan
    configs: list[RunConfig] = []
    for inst_num in instances:
        for tl in time_limits:
            for algo in ALGOS:
                for seed in range(REPEATS):
                    configs.append(
                        RunConfig(
                            algorithm=algo,
                            neighborhood=args.neigh,
                            instance_file=N500_FILE,
                            instance_number=inst_num,
                            seed=seed,
                            time_limit_ms=tl,
                            tabu_tenure=TABU_TENURE if algo == "ils" else None,
                        )
                    )

    print(
        f"[Chunk] neigh={args.neigh}  instances={instances}  "
        f"tl={time_limits}  ({len(configs)} runs)  → {N500_RESULTS_DIR}/"
    )

    runner = ExperimentRunner(
        resume_dir=N500_RESULTS_DIR,
        generate_plots=False,
    )
    runner.run(configs)
    print("[Chunk] done")


if __name__ == "__main__":
    main()
