"""QPU campaign — tranche P5: fill the remaining time-budget layers of
tab:rpd_n20_tl (Table 6, per-budget table at n=20, m=5).

Two targets, each resuming its original campaign directory (completed
runs are skipped via result.json, budget-failed ones retry):

  --target enh   -> results/experiments/20260707_203922 (P2/T3)
                    missing: tl=2000, tl=10000 (all four), tl=100 rest
                    (~914 runs, est. ~85-100 s access -> 2-3 tranches)
  --target orig  -> results/experiments/20260703_083952 (P1)
                    missing: tl=100, 500, 2000, 10000 (+4 at tl=1000)
                    (~1525 runs, est. ~70-85 s access -> 2-3 tranches)

Plan order is neighborhood-major with dynasearch LAST: its minorminer
grind (~2.4 min/run for the full dense QUBO) must not block the cheap
cells. The full grid is enumerated; resume-skip handles what's done.

Usage:
    .venv311/bin/python3 scripts/run_qpu_p5.py --target enh
    .venv311/bin/python3 scripts/run_qpu_p5.py --target orig
    ... --limit 55   # micro-test (first ~50 entries skip as done)

Requires DWAVE_API_TOKEN in the environment (.env). Budget cap comes
from quantum.qpu_budget_s (40 s per launch).
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

INSTANCE_FILE = "data/tai20_5.txt"
TARGETS = {
    "enh": {
        "resume": "results/experiments/20260707_203922",
        "neighborhoods": (
            "quantum_adjacent_enhanced",
            "quantum_fibonacci_enhanced",
            "quantum_motzkin_enhanced",
            "quantum_dynasearch_enhanced",  # wall-heavy last
        ),
    },
    "orig": {
        "resume": "results/experiments/20260703_083952",
        "neighborhoods": (
            "quantum_adjacent",
            "quantum_fibonacci",
            "quantum_motzkin",
            "quantum_dynasearch",  # minorminer grind last
        ),
    },
}
TIME_LIMITS_MS = (100, 500, 1000, 2000, 5000, 10000)  # full grid; done cells skip
ALGOS = ("ils", "sa")
INSTANCES = range(10)
SEEDS = range(5)


def build_plan(neighborhoods) -> list[RunConfig]:
    plan: list[RunConfig] = []
    for neigh in neighborhoods:
        for tl in TIME_LIMITS_MS:
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
    ap.add_argument("--target", choices=("enh", "orig"), required=True)
    ap.add_argument("--limit", type=int, default=None, help="run only first N plan entries")
    args = ap.parse_args()
    tgt = TARGETS[args.target]

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qconf = dict(cfg.get("quantum") or {})
    qconf["backend"] = "dwave"  # explicit opt-in for this campaign script

    print(f"[P5:{args.target}] backend=dwave, qpu_budget_s={qconf.get('qpu_budget_s')}, "
          f"resume={tgt['resume']}")

    runner = ExperimentRunner(quantum_config=qconf, resume_dir=tgt["resume"])
    print(f"[P5:{args.target}] results dir: {runner.timestamp_dir}")
    plan = build_plan(tgt["neighborhoods"])
    if args.limit is not None:
        plan = plan[: args.limit]
    print(f"[P5:{args.target}] planned runs (before resume-skip): {len(plan)}")
    runner.run(plan)
    print(f"[P5:{args.target}] DONE. Results dir: {runner.timestamp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
