"""QPU campaign — P6: close Table 6 budget columns tl=100/500/2000 at n=20
within a HARD 90 s total QPU-access allowance (granted 2026-07-24).

Strategy: tl-major order so whole columns close first; the expensive
tl=10000 layer is deliberately excluded (does not fit in 90 s).

  --target enh   resume results/experiments/20260707_203922
                 layers: tl=100 (stragglers), tl=2000 (all four)
                 est. ~40 s access -> run with --cap 45
  --target orig  resume results/experiments/20260703_083952
                 layers: tl=100, tl=500 (rest), tl=2000
                 est. ~40 s access -> cap = remaining allowance

Run enh first, measure actual usage from results/qpu_timing.jsonl,
then launch orig with cap = 88 - used_enh (2 s safety margin).

Usage:
    .venv311/bin/python3 scripts/run_qpu_p6.py --target enh --cap 45
    .venv311/bin/python3 scripts/run_qpu_p6.py --target orig --cap <adaptive>
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
            "quantum_dynasearch_enhanced",  # wall-heavy last within each tl
        ),
        "tls": (100, 2000),
    },
    "orig": {
        "resume": "results/experiments/20260703_083952",
        "neighborhoods": (
            "quantum_adjacent",
            "quantum_fibonacci",
            "quantum_motzkin",
            "quantum_dynasearch",  # minorminer grind last within each tl
        ),
        "tls": (100, 500, 2000),
    },
}
ALGOS = ("ils", "sa")
INSTANCES = range(10)
SEEDS = range(5)


def build_plan(tgt) -> list[RunConfig]:
    plan: list[RunConfig] = []
    for tl in tgt["tls"]:              # tl-major: close whole columns first
        for neigh in tgt["neighborhoods"]:
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
    ap.add_argument("--cap", type=float, required=True,
                    help="QPU access budget cap for THIS launch [s]")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tgt = TARGETS[args.target]

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    qconf = dict(cfg.get("quantum") or {})
    qconf["backend"] = "dwave"
    qconf["qpu_budget_s"] = args.cap  # hard self-metered guard

    print(f"[P6:{args.target}] backend=dwave, cap={args.cap}s, "
          f"resume={tgt['resume']}, tls={tgt['tls']}")

    runner = ExperimentRunner(quantum_config=qconf, resume_dir=tgt["resume"])
    plan = build_plan(tgt)
    if args.limit is not None:
        plan = plan[: args.limit]
    print(f"[P6:{args.target}] planned (before resume-skip): {len(plan)}")
    runner.run(plan)
    print(f"[P6:{args.target}] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
